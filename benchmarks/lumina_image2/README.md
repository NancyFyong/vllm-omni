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
