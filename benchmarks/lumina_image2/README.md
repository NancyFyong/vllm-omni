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
without ragged padding). Measured with the **official diffusion serving benchmark harness**
(`tests/dfx/perf/scripts/run_diffusion_benchmark.py`, same tool as
[#4995](https://github.com/vllm-project/vllm-omni/pull/4995)) — 512×512, 20 steps, random
dataset, warmup excluded — sweeping client concurrency 1 / 2 / 4 for a serial engine
(`--max-num-seqs 1`) vs. a batching engine (`--max-num-seqs 4`):

| Mode | Concurrency | Prompts | Throughput (qps) | Latency mean (s) | p99 (s) | Peak mem (MB) | Success |
|---|---|---|---|---|---|---|---|
| serial (`max_num_seqs=1`) | 1 | 4 | 0.478 | 2.091 | 2.124 | 11544 | 4/4 |
| serial (`max_num_seqs=1`) | 2 | 8 | 0.501 | 3.747 | 4.092 | 11544 | 8/8 |
| serial (`max_num_seqs=1`) | 4 | 16 | 0.497 | 7.303 | 8.139 | 11544 | 16/16 |
| batch (`max_num_seqs=4`) | 1 | 4 | 0.476 | 2.099 | 2.107 | 12466 | 4/4 |
| batch (`max_num_seqs=4`) | 2 | 8 | 0.503 | 3.725 | 4.034 | 12466 | 8/8 |
| batch (`max_num_seqs=4`) | 4 | 16 | **0.522** | 7.321 | 9.098 | 13812 | 16/16 |

Fusion is real — logs show `[RequestBatch] admission wait done waiting=4 max_batch=4`, and
peak memory grows with the fused batch (11544 → 13812 MB) as four images denoise together.
Forcing a full batch of 4 with `--request-batch-max-wait-ms 300` (so the engine waits to
gather 4 requests before each forward) gives **0.514 qps, 7.787 s mean, 9.227 s p99,
14748 MB peak** at concurrency 4 — again on par with serial.

The honest read: on Lumina2 at 512² the Next-DiT forward is roughly **compute-bound** on
H20, so fusing four requests into one larger GEMM keeps throughput essentially flat (batch
is within ±5% of serial, and marginally *higher* at concurrency 4: 0.522 vs 0.497 qps) —
it does **not** regress. It trades a modest memory increase for equal-to-slightly-better
throughput and does not help mean latency (the images finish together). For a real speedup
on this model, scale out with the parallelism flags above (TP / CFG / Cache-DiT); request
batching is kept for correctness parity and for headroom on smaller requests. This matches
why other recent single-DiT image pipelines ship serial-by-default
([#4995](https://github.com/vllm-project/vllm-omni/pull/4995); `z_image`, `longcat_image`,
`ovis_image`, `hunyuan_image3`).

Sample `Serving Benchmark Result` block (serial vs. batch at concurrency 4):

```text
============ Serving Benchmark Result ============   ← serial, max_num_seqs=1
Max request concurrency:                 4
Successful requests:                     16/16
Request throughput (req/s):              0.50
Latency Mean (s):                        7.3027
Latency P99 (s):                         8.1391
Peak Memory Max (MB):                    11544.00
==================================================

============ Serving Benchmark Result ============   ← batch, max_num_seqs=4
Max request concurrency:                 4
Successful requests:                     16/16
Request throughput (req/s):              0.52
Latency Mean (s):                        7.3212
Latency P99 (s):                         9.0984
Peak Memory Max (MB):                    13812.00
==================================================
```

Reproduce (official harness, config committed under `tests/dfx/perf/tests/`):

```bash
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_vllm_omni.json -s -v
# forced full batch=4 (adds --request-batch-max-wait-ms 300):
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_forcedbatch.json -s -v
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
