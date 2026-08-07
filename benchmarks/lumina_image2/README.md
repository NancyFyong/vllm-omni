# Lumina-Image-2.0 Benchmarks

Benchmarks for Lumina-Image-2.0 (Next-DiT) text-to-image across three backends:
diffusers baseline, vLLM-Omni offline, and vLLM-Omni online serving.

| Benchmark | Script | Description |
|-----------|--------|-------------|
| diffusers baseline | `huggingface/inference.py` | Single-GPU `Lumina2Pipeline` reference (images + speed) |
| vLLM-Omni offline | `vllm-omni/inference.py` | Offline inference, same prompts/seed/size/steps |
| vLLM-Omni online | official `benchmarks/diffusion/diffusion_benchmark_serving.py` | `/v1/chat/completions`, latency/throughput/stage timing |

All runs: **1024×1024, 30 steps, guidance 4.0, seed 42**, bf16, on **NVIDIA H20**,
torch 2.11.0+cu130, diffusers 0.38.0. Warmup excluded.

## Quality: diffusers vs vLLM-Omni

Side-by-side for 4 prompts (landscape / portrait / text-in-image / art). Outputs
are visually near-identical; both render in-image text ("OPEN") correctly.

![diffusers vs vLLM-Omni comparison grid](https://raw.githubusercontent.com/NancyFyong/vllm-omni/lumina-image2-assets/assets/lumina-image2/comparison.png)

Full-resolution per-prompt outputs (diffusers and vLLM-Omni) are hosted on the
[`lumina-image2-assets`](https://github.com/NancyFyong/vllm-omni/tree/lumina-image2-assets/assets/lumina-image2)
branch.

## Speed

### Single-GPU latency (per image, mean over 4 prompts)

| Backend | Latency (ms) | Per-step (ms) | Speedup vs diffusers |
|---|---|---|---|
| diffusers (`Lumina2Pipeline`) | 14416 | 480.6 | 1.00× |
| vLLM-Omni offline | 10910 | 363.7 | **1.32×** |
| vLLM-Omni online (serving) | 11224 | 374.1 | 1.28× |

Serving latency includes end-to-end request overhead (text-encode 40 ms +
VAE-decode 172 ms + queue 0.2 ms + DiT 10888 ms); peak GPU memory 15014 MB.

### Parallelism / acceleration (vLLM-Omni serving, H20)

| Config | Latency mean (ms) | Median (ms) | p99 (ms) | Throughput (qps) | Notes |
|---|---|---|---|---|---|
| base (1 GPU) | 11224 | 11258 | 11321 | 0.089 | reference |
| TP=2 | 6716 | 6713 | 6839 | 0.149 | tensor parallel, **1.67×** |
| SP=2 (Ulysses) | 7313 | 7309 | 7529 | 0.137 | sequence parallel, **1.53×** |
| CFG=2 | 5959 | 5955 | 6078 | 0.168 | CFG-parallel, **1.88×** |
| Cache-DiT | 5647 | 5686 | 5804 | 0.177 | 1 GPU + cache, **1.99×** |
| HSDP (shard=2) | 12001 | 12005 | 12101 | 0.083 | param-sharding: 14350 MB/GPU peak vs 15014 MB base (memory-scaling, not latency) |

### Request-level batching (1 GPU, H20)

Lumina2 supports request-level fused batching (`supports_request_batch=True`): with
`--max-num-seqs > 1` the engine accumulates concurrent requests and runs them as one fused
Next-DiT batch (Gemma pads every prompt to a fixed length, so per-request embeddings stack
without ragged padding). This is genuine fusion — the worker calls `pipeline.forward` once
on a `DiffusionRequestBatch`, and logs confirm `[RequestBatch] admission wait done
waiting=4 max_batch=4`.

**Does it help throughput? No — it is throughput-neutral at this workload.** Measured with
the **official diffusion serving benchmark harness**
(`tests/dfx/perf/scripts/run_diffusion_benchmark.py`, same tool as
[#4995](https://github.com/vllm-project/vllm-omni/pull/4995)) — 512×512, 20 steps, random
dataset, negative prompt on, concurrency 4, 16 prompts, warmup excluded, **3 measured
repeats each** for mean ± spread:

| Engine mode | Throughput (qps), 3 runs | Mean qps | Latency mean (s) | Peak mem (MB) |
|---|---|---|---|---|
| serial (`max_num_seqs=1`) | 0.506 / 0.502 / 0.506 | **0.504** | 7.19 | 11544 |
| opportunistic batch (`max_num_seqs=4`) | 0.451 / 0.453 / 0.521 | 0.475 | 8.12 | 13758 |
| forced full batch (`max_num_seqs=4 --request-batch-max-wait-ms 300`) | 0.514 / 0.509 / 0.507 | **0.510** | 7.84 | 15160 |

The honest read, backed by the numbers above:

- **Serial is tight and fast** (0.504 qps, ±0.002 across runs).
- **Forced full-batch matches serial** — 0.510 vs 0.504 qps is +1.2%, inside run-to-run
  spread — while costing **+31% peak memory** (15160 vs 11544 MB) and **higher mean
  latency** (7.84 vs 7.19 s, because the four images finish together). Not a win.
- **Opportunistic batching (no `--request-batch-max-wait-ms`) is the *worst* option**: its
  throughput is wide and unstable (0.451–0.521) because it fuses whatever happens to be
  queued. When it genuinely fuses a batch of four the per-batch forward gets larger and
  end-to-end throughput drops to ~0.45; when arrivals miss the window it degenerates to
  serial. It adds variance and memory with no upside.

**Why fusion cannot help here:** the diffusion (denoise) stage is **≈98% of end-to-end
latency** in every arm (`stage_0_gen_ms` ≈ 7.0–7.4 s vs `text_encoder` 42 ms + `vae.decode`
43 ms), and at 512² a single-image Next-DiT forward already saturates the H20. Fusing four
requests into one forward does not reduce total FLOPs and there is no spare GPU utilization
to reclaim, so throughput stays flat while activation memory and (with CFG) the effective
batch grow. For a real speedup on this model, scale out with the parallelism flags above
(TP / CFG / Cache-DiT). Request batching is kept for correctness parity and for headroom on
*smaller* requests, matching why other recent single-DiT image pipelines ship
serial-by-default ([#4995](https://github.com/vllm-project/vllm-omni/pull/4995);
`z_image`, `longcat_image`, `ovis_image`, `hunyuan_image3`).

**Correctness of the fused path (merge gate).** Four fixed `(prompt, seed)` requests sent
serial (batch-of-1) vs. concurrent (fused batch=4) through one server, same seeds,
512²/20 steps. Both paths are individually **bit-exact on rerun** (serial-vs-serial and
fused-vs-fused PSNR = ∞), so any serial-vs-fused difference is deterministic, not sampling
noise, and seed/prompt routing is verified correct. Serial vs. fused: **3 of 4 prompts
near-identical (40.4 / 41.7 / 42.8 dB PSNR, SSIM ≥ 0.99); 1 of 4** (a high-frequency
portrait) is the **same subject/pose/composition** but with perturbed fine texture (20.2 dB,
SSIM 0.80). The cause is the bf16 batched-GEMM reduction-order change amplified over the
20-step Euler sampler — the standard "numerically-equivalent-in-expectation, not bit-exact"
behavior of fused diffusion batching, not a routing or correctness bug.

Reproduce (official harness, configs committed under `tests/dfx/perf/tests/`):

```bash
# serial vs opportunistic batch, concurrency 4 (run 3x for spread):
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_c4_repeat.json -s -v
# forced full batch=4 (adds --request-batch-max-wait-ms 300):
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_c4_forcedbatch_repeat.json -s -v
```

## Reproduce

```bash
# diffusers baseline
CUDA_VISIBLE_DEVICES=0 python benchmarks/lumina_image2/huggingface/inference.py

# vLLM-Omni offline
CUDA_VISIBLE_DEVICES=0 python benchmarks/lumina_image2/vllm-omni/inference.py

# vLLM-Omni online serving
vllm serve Alpha-VLLM/Lumina-Image-2.0 --omni --port 8091 \
    --enable-diffusion-pipeline-profiler
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --endpoint /v1/chat/completions --task t2i --dataset random \
    --num-prompts 8 --num-inference-steps 30 --height 1024 --width 1024 \
    --warmup-requests 1 --max-concurrency 1 --port 8091

# comparison grid
python benchmarks/lumina_image2/make_comparison.py \
    --diffusers benchmarks/lumina_image2/outputs/diffusers \
    --vllm benchmarks/lumina_image2/outputs/vllm_omni
```

## Parallelism / acceleration flags

Append to `vllm serve ... --omni` and re-run the same serving benchmark:

```bash
--tensor-parallel-size 2                      # TP=2
--ulysses-degree 2                            # SP=2 (Ulysses)
--cfg-parallel-size 2                         # CFG-parallel
--cache-backend cache_dit                     # Cache-DiT (1 GPU)
--use-hsdp --hsdp-shard-size 2 --hsdp-replicate-size 1   # HSDP param-sharding
--enable-cpu-offload                          # layerwise CPU offload
```
