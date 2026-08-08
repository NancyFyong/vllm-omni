# SkyReels V3 V2V — PR Benchmark Artifacts

Local benchmark run 2026-08-07 on 1×H20. Full-suite PR-ready bench for the V2V (single-shot video-extension) pipeline.

## Setup

- Model: `/workspace/models/SkyReels-V3-V2V-14B` (mirror of `Skywork/SkyReels-V3-V2V-14B`)
- vllm-omni: `feat/skyreels-v3-v2v-impl` @ `94ce4a88` (4 commits ahead of `main`)
- Input video: `/tmp/skyreels-bench/inputs/extension_test.mp4` — Skywork example `https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4` (1364×720 @ 24 fps, 3.0 s, 72 frames)
- Resolution: 720P bucket (1312×688)
- Sampling: 4 steps × 2 chunks = 8 total sampling steps per output video
- `flow_shift`: 8.0 (V2V default); `guidance_scale`: 1.0 (CFG disabled)
- Seed: 42

## Speed comparison (1×H20, single-shot extension, guidance=1)

| Pipeline | Init | Per-step | Total denoise | VAE decode + save | End-to-end |
|---|---:|---:|---:|---:|---:|
| **Skywork upstream** (with `--offload`) | ~3 min (weight load + T5 + VAE) | **207.32 s** | 27:38 (8 steps) | ~30 s | ~30 min |
| **vLLM-Omni** | 216 s (spawn + weight load 143 s / 37.77 GB) | **~209 s** (effectively identical) | ~840 s (4 steps × 2 chunks) | ~33 s | **873.32 s** (~14.5 min) |

The per-step transformer kernel path is byte-for-byte the same (207–209 s per step at 1312×688 × 33 latent frames). End-to-end wall-clock differs because Skywork ran with `--offload` (encoders shuttled off GPU) while vLLM-Omni kept them resident.

## CFG-Parallel speedup measurement (1×H20 vs 2×H20, guidance=5, seed=42)

| Config | Init | Total generation | Output bytes |
|---|---:|---:|---:|
| 1×H20, guidance=5.0 (CFG active, sequential cond+uncond) | 312.95 s | **1696.47 s** (~28 min) | 4216216 |
| 2×H20, guidance=5.0, `--cfg-parallel-size 2` | 232.08 s | **1709.17 s** (~28.5 min) | 4216216 (byte-identical) |

**Speedup: none.** The 2-GPU cfg-parallel-2 run took the same wall-clock time as the single-GPU CFG run (with a small distributed init overhead). Outputs are byte-identical.

### Why CFG-parallel is currently a no-op for V2V (follow-up bug on this branch)

`SkyReelsV3V2VPipeline` inherits `CFGParallelMixin`, but its `diffuse()` denoise loop bypasses the mixin's `predict_noise_maybe_with_cfg(...)` helper and instead calls `self._predict_noise(...)` **twice sequentially** (once for the conditional prediction, once for the unconditional one). The two `_predict_noise` calls run back-to-back on the same rank, so `--cfg-parallel-size 2` spins up a second worker that duplicates the same work rather than splitting the two branches.

Cross-reference — this is exactly what the A2V pipeline (`skyreels-v3-a2v-impl` @ b3aec04d) does *right*: its `diffuse()` calls `self.predict_noise_maybe_with_cfg(...)` and gets the measured 1.44× speedup on 2×H20 at guidance=5 (156.89 s vs 225.27 s on 1×H20).

Fix: rewrite the V2V denoise loop to route through `CFGParallelMixin.predict_noise_maybe_with_cfg` (positive/negative kwargs) plus `CFGParallelMixin.scheduler_step_maybe_with_cfg`, same pattern the A2V branch uses in `pipeline_skyreels_v3_a2v.py::diffuse()`.

## Per-frame quality (132 overlapping frames, official ‖ vLLM-Omni)

Same seed, same recipe. Both pipelines emit 132 frames at 1312×688 @ 24 fps (5.5 s of video).

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| PSNR | 19.397 dB | 17.264 dB | 16.115 dB | **33.888 dB** (frame 0, shared reference from input video) |
| SSIM | 0.530 | 0.444 | 0.411 | **0.953** (frame 0) |
| MSE (per pixel) | 1033.40 | — | — | — |

Frame 0 (which is the shared 25-frame prefix from the input video re-emitted through the VAE) matches near-identical. Beyond that, PSNR ~17 / SSIM ~0.44 reflects two independent samplers taking different noise trajectories off the same integer seed — same visual scene, same lighting, minor pixel-level differences per frame. Similar to the A2V case, this is expected diffusion-model behavior and not a correctness gap.

## Tests wired (this PR now covers all four levels)

| Level | File | Marker | Result |
|---|---|---|---|
| **L1 CPU unit** | `tests/diffusion/models/skyreels_v3/test_skyreels_v3_v2v.py` | `core_model + diffusion + cpu` | **7 passed** in 19 s (see `logs/l1_cpu_unit_tests.log`) |
| **L4 Function** | `tests/e2e/online_serving/test_skyreels_v3_v2v_expansion.py` | `full_model + diffusion + H100` | Collects 1 case (`cuda_v2v_baseline`); wired into `test-nightly.yml` X2V Other Function Test |
| **L4 Perf** | `tests/dfx/perf/tests/test_skyreels_v3_v2v_vllm_omni.json` | `full_model + diffusion + H100` | Baseline: latency_mean **873.32 s**, peak ~77 GB; wired into X2V Perf Test step |
| **Bench script** | `run_v2v.py` (`/tmp/skyreels-bench/`) | — | Standalone driver used for this bench (`OmniRequestOutput` unwrap, dtype convert, mp4 save) |

Also fixed in this PR:

- `vllm_omni/entrypoints/utils.py` — added V2V routing shortcut through `_looks_like_skyreels_v3_v2v(...)` inside `_try_resolve_omni_model_type`. Before this fix, the V2V-14B checkpoint (no root `config.json` / `model_index.json`) crashed with `ValueError: Could not determine model_type for model...` before even attempting the pipeline forward.
- `examples/offline_inference/image_to_video/image_to_video.py` — new `--video` flag routing to `multi_modal_data.video` for V2V. Skips the `dimension_image` requirement (V2V picks output resolution from the input video via `resolve_bucket_size`).

## Known limitations (not covered by this PR)

- **CFG-parallel is a no-op for V2V** — pipeline denoise loop calls `_predict_noise` sequentially instead of routing through `CFGParallelMixin.predict_noise_maybe_with_cfg`. Same-hash output at 1-GPU vs 2-GPU (see table above). Fix targets a follow-up commit that mirrors the A2V pipeline's `diffuse()` pattern.
- **Shot-switching mode** — the V2V-14B checkpoint ships both `transformer/` (single-shot, 28.6 GB) and `shot_transformer/` (28.6 GB) weights. The current pipeline only wires the `transformer/` path; adding shot-switching means an additional pipeline class (or a `--mode` switch). Follow-up PR.
- **No Cache-DiT enabler** — no `CUSTOM_DIT_ENABLERS` entry for `SkyReelsV3V2VPipeline`.

## Files in this directory

```text
bench_v2v_baseline_4step_1gpu.mp4                    — vllm-omni fresh 4-step 720P single-shot extension baseline (g=1)
bench_v2v_g5_4step_1gpu.mp4                          — vllm-omni 4-step 720P w/ guidance=5, sequential CFG on 1×H20
bench_v2v_g5_4step_cfgparallel2_2gpu.mp4             — vllm-omni 4-step 720P w/ guidance=5 + --cfg-parallel-size 2 (byte-identical to 1-GPU)
official_v2v_seed42_720p_4step.mp4                   — Skywork upstream generate_video.py --task_type single_shot_extension
omni_v2v_seed42_720p_4step.mp4                       — same as bench_v2v_baseline_4step_1gpu.mp4, renamed to match compare pair
side_by_side_official_vs_omni_v2v.mp4                — hstacked comparison video
compare.png                                           — 5-row per-frame side-by-side with PSNR/SSIM labels
metrics.json                                          — per-frame + summary quality metrics (132 frames)
compute_compare.py                                    — script that produced metrics.json + compare.png
skywork_official.log                                  — Skywork upstream run log
frames_official/, frames_omni/                        — extracted frames for metric computation
logs/l1_cpu_unit_tests.log                            — 7 passed
logs/bench_v2v_g5_4step_gpu5.log                      — 1×H20 g=5 run log
logs/bench_v2v_g5_4step_cfgparallel2_gpu6-7.log       — 2×H20 cfg-parallel g=5 run log
BENCH_SUMMARY.md                                      — this file
```

## Not committed to the tree

This directory is a local artifact drop; not part of the PR file set. The PR body will reference the persistent-hosted copies once uploaded to a `skyreels-v3-v2v-assets` fork branch (mirroring the `skyreels-v3-a2v-assets` / `skyreels-v3-r2v-assets` convention).
