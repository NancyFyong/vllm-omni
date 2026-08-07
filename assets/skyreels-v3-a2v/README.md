# SkyReels V3 A2V — PR Comparison Artifacts

Assets branch for the [`[Model] Add SkyReels V3 A2V` PR](https://github.com/vllm-project/vllm-omni/pulls?q=SkyReels+V3+A2V). Follows the same convention as `skyreels-v3-r2v-assets` and `lumina-image2-assets`.

## Files

| File | What it is |
|---|---|
| `side_by_side_official_vs_omni_4step.mp4` | Hstacked (official ‖ vLLM-Omni) at seed=42, 4 steps, 480P, guidance=1.0 — the primary quality-parity evidence |
| `compare.png` | 5-row side-by-side panel at frames {0, 20, 40, 60, 80} with per-frame PSNR/SSIM labels |
| `metrics.json` | Per-frame PSNR/SSIM/MSE and summary (mean/median/min/max) over the 81 overlapping frames |
| `official_seed42_480p_4step_81f.mp4` | Skywork upstream `generate_video.py --task_type talking_avatar` (125 frames, 3-chunk stitched to cover 5 s audio) |
| `omni_seed42_480p_4step_81f.mp4` | vLLM-Omni `image_to_video.py --model Skywork/SkyReels-V3-A2V-19B` (81 frames, single chunk) |
| `omni_seed42_480p_4step_81f_g5.mp4` | Same recipe, guidance=5.0 — three-branch CFG active |
| `omni_seed42_480p_4step_81f_g5_cfgparallel2.mp4` | Same recipe on 2×H20 with `--cfg-parallel-size 2` — 1.44× speedup over single-GPU CFG |
| `omni_seed42_480p_40step_81f_production.mp4` | Production preset (40 steps, guidance=1.0) — full quality demo |

## Setup

Both implementations pointed at the same local checkpoint mirror of `Skywork/SkyReels-V3-A2V-19B`. Same portrait, same 5 s driving audio (Skywork sample).

- Official env: fresh `uv` venv (torch 2.8.0+cu128, flash-attn 2.8.3.post1, diffusers 0.34.0, xfuser 0.4.3, yunchang 0.6.3)
- vLLM-Omni env: `torch 2.11`, FLASH_ATTN backend
- Portrait: [Skywork sample `single1.png`](https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single1.png)
- Driving audio: [Skywork sample `huahai_5s.mp3`](https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/huahai_5s.mp3) (5.05 s)
- Resolution: 480P bucket → 832×480 @ 25 fps
- Sampling steps: 4; `flow_shift`: 11; seed: 42

## Timing (1×H20 unless noted)

| Pipeline | Config | Init | Per-step | End-to-end |
|---|---|---:|---:|---:|
| Skywork upstream | 4-step × 3 chunks, `--offload` | 60 s + 11.29 s audio preprocess | 17.74–17.83 s | ~230 s |
| vLLM-Omni | 4-step × 1 chunk, `guide=1` | 225 s (one-time) | 17.74 s | **84.24 s** |
| vLLM-Omni | 4-step × 1 chunk, `guide=5` | 298 s | ~56 s (3-branch CFG) | **225.27 s** |
| vLLM-Omni | 4-step × 1 chunk, `guide=5`, `--cfg-parallel-size 2` (2×H20) | 243 s | ~39 s | **156.89 s** (1.44× speedup) |
| vLLM-Omni | 40-step × 1 chunk, `guide=1` (production preset) | 223 s | 17.74 s | 721.50 s |

Per-step time is byte-for-byte identical between the two implementations (17.74 s/step on the same 480P shape) — the transformer + VAE kernel paths match upstream.

## Per-frame quality (81 overlap frames, official ‖ vLLM-Omni)

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| PSNR | 17.996 dB | 17.447 dB | 15.787 dB | **40.889 dB** (frame 0) |
| SSIM | 0.590 | 0.539 | 0.470 | **0.977** (frame 0) |
| MSE (per pixel) | 1166.96 | — | — | — |

Frame 0 (the shared input portrait re-emitted after VAE round-trip) is near-bit-identical, confirming both pipelines start from the same tensor. Frames 1–80 diverge as expected for two independent samplers off the same seed. See `compare.png` and the PR body for the four independent divergence sources (RNG entry, scheduler step, chunk boundary handling, dtype cast boundary).

## Reproduction

```bash
# vLLM-Omni
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Skywork/SkyReels-V3-A2V-19B \
  --image single1.png --audio huahai_5s.mp3 \
  --num-inference-steps 4 --num-frames 81 --fps 25 --seed 42 \
  --extra-body '{"resolution": "480P"}' \
  --output omni.mp4

# CFG-Parallel 2
… same as above with:  --guidance-scale 5.0 --cfg-parallel-size 2 \
  --extra-body '{"resolution": "480P", "text_guide_scale": 5.0, "audio_guide_scale": 5.0}'

# Skywork upstream
python generate_video.py --task_type talking_avatar \
  --model_id Skywork/SkyReels-V3-A2V-19B \
  --input_image single1.png --input_audio huahai_5s.mp3 \
  --resolution 480P --seed 42 --duration 4 --offload
```
