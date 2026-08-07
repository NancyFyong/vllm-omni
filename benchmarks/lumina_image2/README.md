# Lumina-Image-2.0 Benchmarks

Benchmarks for Lumina-Image-2.0 (Next-DiT) — text-to-image across three backends
(diffusers baseline, vLLM-Omni offline, vLLM-Omni online serving), plus
SDEdit-style **image-to-image** on the same served checkpoint.

| Benchmark | Script | Description |
|-----------|--------|-------------|
| diffusers baseline (t2i) | `huggingface/inference.py` | Single-GPU `Lumina2Pipeline` reference (images + speed) |
| vLLM-Omni offline (t2i) | `vllm-omni/inference.py` | Offline inference, same prompts/seed/size/steps |
| vLLM-Omni offline (i2i) | `vllm-omni/inference_i2i.py` | Same served model, SDEdit i2i sweeping `strength` |
| vLLM-Omni online (t2i / i2i) | official `benchmarks/diffusion/diffusion_benchmark_serving.py` | `/v1/chat/completions`, latency/throughput/stage timing |

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

## Image-to-image (SDEdit)

Lumina-Image-2.0 ships no editing checkpoint — image-to-image runs SDEdit-style
on the same t2i weights: the input image is VAE-encoded, re-noised at the sigma
selected by `strength`, and only the tail of the schedule is denoised. A request
carrying `multi_modal_data["image"]` takes the i2i path; everything else is
unchanged. See [../../examples/offline_inference/image_to_image/image_to_image.md](../../examples/offline_inference/image_to_image/image_to_image.md).

No diffusers Lumina2 img2img pipeline exists, so a like-for-like external
baseline is omitted; the numbers below characterize how the served pipeline
behaves under concurrency, request-level batching, and the same
parallelism/acceleration flags the t2i table uses, using the same official
harness.

### Parallelism / acceleration (vLLM-Omni serving, H20, i2i)

Same axes as the t2i "Parallelism / acceleration" table (1024×1024, 30 steps,
concurrency 1), plus **strength = 0.6** so the DiT sees ~18 residual steps
after SDEdit re-noises the input.

| Config | Latency mean (ms) | p99 (ms) | Throughput (qps) | Peak mem (MB) | Notes |
|---|---|---|---|---|---|
| base (1 GPU) | 6905 | 6927 | 0.14 | 14628 | reference |
| TP=2 | 4203 | 4214 | 0.24 | 12488 | tensor parallel, **1.64×** |
| SP=2 (Ulysses) | 4832 | 4847 | 0.21 | 14724 | sequence parallel, **1.43×** |
| CFG=2 | 3707 | 3776 | 0.27 | 14722 | CFG-parallel, **1.86×** |
| Cache-DiT | 4110 | 4189 | 0.24 | 14628 | 1 GPU + cache, **1.68×** |

The speedups track the t2i table almost 1-to-1 — CFG=2 wins because it doubles
the two prompt branches across cards, TP=2 wins because it splits the DiT
weight/attention across cards, both close to their t2i ratios (i2i 1.86× /
1.64× vs. t2i 1.88× / 1.67×). Cache-DiT drops from 1.99× on t2i to 1.68× on
i2i because SDEdit runs fewer steps to cache across (~18 vs. 30). Ulysses
tracks similarly (i2i 1.43× vs. t2i 1.53×).

**TeaCache is not applicable** to Lumina2 today: `--cache-backend tea_cache`
raises `KeyError: Cannot find coefficients for Lumina2Transformer2DModel`, and
no rescaling coefficients have been fit upstream yet (supported so far: Flux,
Qwen-Image, Z-Image, LongCat-Image, etc.). Cache-DiT is the applicable cache
backend for now.

### Request-level batching (1 GPU, H20, i2i)

Same harness (`tests/dfx/perf/scripts/run_diffusion_benchmark.py`), same
methodology as the t2i table above — 512×512, 20 steps, **strength = 0.6**
(denoises the tail ~60% of the schedule), random dataset with synthetic input
images, warmup excluded. Sweeping client concurrency 1 / 2 / 4 for a serial
engine (`--max-num-seqs 1`) vs. a batching engine (`--max-num-seqs 4`):

| Mode | Concurrency | Prompts | Throughput (qps) | Latency mean (s) | p99 (s) | Peak mem (MB) | Success |
|---|---|---|---|---|---|---|---|
| serial (`max_num_seqs=1`) | 1 | 4 | 0.750 | 1.335 | 1.346 | 11544 | 4/4 |
| serial (`max_num_seqs=1`) | 2 | 8 | 0.786 | 2.377 | 2.565 | 11544 | 8/8 |
| serial (`max_num_seqs=1`) | 4 | 16 | 0.788 | 4.568 | 5.068 | 11544 | 16/16 |
| batch (`max_num_seqs=4`) | 1 | 4 | 0.767 | 1.301 | 1.310 | 12466 | 4/4 |
| batch (`max_num_seqs=4`) | 2 | 8 | 0.785 | 2.386 | 2.584 | 12466 | 8/8 |
| batch (`max_num_seqs=4`) | 4 | 16 | **0.812** | 4.763 | 6.227 | 13812 | 16/16 |

Two things worth noting up front. First, i2i mean latency at `strength=0.6` is
roughly **63% of t2i** at the same size/steps (1.335 s vs 2.091 s at c=1) — SDEdit
skips the first ~40% of the denoise, and that ratio holds all the way up the
concurrency sweep. Second, request-batch fusion is again real for i2i — peak
memory at concurrency 4 grows from 11544 → 13812 MB as four images denoise
together, and `[RequestBatch] admission wait done waiting=4 max_batch=4`
appears in the batch server log.

The batching contract in this branch is stricter for i2i than for t2i: fused
requests must additionally share `strength` and whether they carry an input
image, and (when the request gives no `--height`/`--width`) they must land on
the same image-derived size. Two i2i requests with mismatched strength or
aspect ratio are explicitly rejected rather than silently resized — the same
philosophy as the existing height/width/steps/guidance homogeneity check.

Forcing a full batch of 4 with `--request-batch-max-wait-ms 300` (so the
engine waits to gather 4 requests before each forward) gives **0.820 qps,
4.892 s mean, 6.184 s p99, 14748 MB peak** at concurrency 4 — on par with
serial and opportunistic batch, and the batch-4 admission logs confirm every
forward fuses 4 requests (`waited_ms` under 4 ms in all five iterations).

The honest read matches t2i: at 512² the Next-DiT forward is compute-bound on
H20 even with only ~12 residual steps, so fusing four requests keeps
throughput essentially flat (batch is within ±5% of serial, and marginally
higher at concurrency 4: 0.812 vs 0.788 qps) and does not help mean latency
(the fused images finish together). Fusion is kept for correctness parity and
smaller-request headroom; for a real speedup, scale out with the same
parallelism flags the t2i section uses (TP / CFG / Cache-DiT).

Sample `Serving Benchmark Result` block (serial vs. batch at concurrency 4, i2i):

```text
============ Serving Benchmark Result ============   ← serial, max_num_seqs=1
Max request concurrency:                 4
Successful requests:                     16/16
Request throughput (req/s):              0.79
Latency Mean (s):                        4.5682
Latency P99 (s):                         5.0678
Peak Memory Max (MB):                    11544.00
==================================================

============ Serving Benchmark Result ============   ← batch, max_num_seqs=4
Max request concurrency:                 4
Successful requests:                     16/16
Request throughput (req/s):              0.81
Latency Mean (s):                        4.7633
Latency P99 (s):                         6.2268
Peak Memory Max (MB):                    13812.00
==================================================
```

Reproduce (official harness, i2i configs committed alongside the t2i ones):

```bash
# request-batch sweep (serial vs. max_num_seqs=4, concurrency 1/2/4)
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_i2i_vllm_omni.json -s -v
# forced full batch=4 (adds --request-batch-max-wait-ms 300):
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_i2i_forcedbatch.json -s -v
# parallelism / acceleration (base, Cache-DiT, and — with 2 visible GPUs — TP=2 / SP=2 / CFG=2):
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_i2i_accel_1gpu.json -s -v
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
    --test-config-file tests/dfx/perf/tests/test_lumina_image2_i2i_accel_2gpu.json -s -v
```

## Reproduce

```bash
# diffusers baseline
CUDA_VISIBLE_DEVICES=0 python benchmarks/lumina_image2/huggingface/inference.py

# vLLM-Omni offline
CUDA_VISIBLE_DEVICES=0 python benchmarks/lumina_image2/vllm-omni/inference.py

# vLLM-Omni offline i2i (SDEdit)
CUDA_VISIBLE_DEVICES=0 python benchmarks/lumina_image2/vllm-omni/inference_i2i.py \
    --input-image benchmarks/lumina_image2/outputs/vllm_omni/landscape.png

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
