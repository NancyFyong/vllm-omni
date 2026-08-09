# Video-To-Video

Instruction-driven video editing: given a source clip and a text instruction, produce an edited clip
that preserves the source's motion and layout.

- `download_joyai_video_edit.py`: fetch the weights and synthesize `model_index.json`.
- `video_to_video.py`: command-line script for a single edit.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [Geometry Constraints](#geometry-constraints)
- [FAQ](#faq)

## Overview

### Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance | VRAM Notes |
| ----- | ------------------ | -------------- | ------------- | -------- | ---------- |
| `jdopensource/JoyAI-Video-Edit` | 720 x 1248 | longest valid prefix of the source | 2 | none (distilled) | ~50 GiB BF16 weights resident; ~51 GiB peak at 720x1248 |

JoyAI-Video-Edit is a 16.2B dual-stream MMDiT paired with a causal video VAE and a MiMo-VL-7B
condition encoder, AR-DMD-distilled down to **two** denoise steps. It generates
**chunk-autoregressively** — one latent frame per step, each attending to a bounded window over the
chunks already generated — so cost scales linearly in clip length with flat memory.

!!! info
    Peak VRAM: single card, batch size 1, no acceleration features. Weights alone are ~47 GiB
    (30 GiB DiT + 15 GiB condition encoder + 1.4 GiB VAE), so this wants an 80 GiB-class card.

## Prerequisites

```bash
python download_joyai_video_edit.py --output-dir /path/to/models/joyai_video_edit
```

That fetches ~47 GiB across two Hugging Face repos (`jdopensource/JoyAI-Video-Edit` and
`XiaomiMiMo/MiMo-VL-7B-RL-2508`) and writes the `model_index.json` that vLLM-Omni dispatches on.
`--model` must be this local directory — a repo id will not work, because the DiT is a raw `.pth`
rather than a diffusers subfolder.

Reading and writing video needs PyAV:

```bash
pip install av
```

## Quick Start

### Command Line

```bash
python video_to_video.py \
    --model /path/to/models/joyai_video_edit \
    --video input.mp4 \
    --prompt "make it snow" \
    --output edited.mp4
```

### Python API

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="/path/to/models/joyai_video_edit", model_class_name="JoyAIVideoEditPipeline")

output = omni.generate(
    {"prompt": "make it snow", "multi_modal_data": {"video": "input.mp4"}},
    OmniDiffusionSamplingParams(height=720, width=1248, num_frames=73, num_inference_steps=2, seed=42),
)
frames = output[0].images[0]  # uint8 [T, H, W, 3]
fps = output[0].multimodal_output["fps"]
```

The source video may be a path, a URL, a `data:video` URL, a numpy array, a torch tensor, or a list
of PIL frames.

## Key Arguments

| Argument | Default | Notes |
| -------- | ------- | ----- |
| `--video` | required | Source clip to edit. This model edits an existing clip and cannot extend one, so the output is never longer than the input. |
| `--prompt` | required | The edit instruction, e.g. `"make it snow"`. There is no negative prompt — the model was distilled without classifier-free guidance. |
| `--num-frames` | longest valid prefix | Must be `1 + 8n`. |
| `--height` / `--width` | 720 / 1248 | Both must be divisible by 24. |
| `--fps` | source video's rate | Output tagging only; does not affect generation. |
| `--seed` | 42 | Seeds both the per-chunk noise and the VAE's source sampling. |

A `num_frames` of 0 **or 1** means "edit the longest valid prefix of the source". The 1 is not a typo:
`OmniDiffusionSamplingParams.num_frames` defaults to `1` for image models, so a caller who never
mentions a frame count is indistinguishable from one asking for a single frame. Taking that literally
would return a one-frame clip from every unparameterised request, so the pipeline reinterprets it and
logs that it did — which also means there is no way to ask for a 1-frame edit of a longer clip.

`num_inference_steps` is fixed at 2 and is not exposed as a knob: the AR-DMD distillation collapsed
the sampling trajectory into exactly two Euler steps on a shift-5.159 sigma grid, so raising it
integrates a velocity field that no longer matches the grid. The pipeline warns rather than silently
degrading.

## Geometry Constraints

Both constraints are enforced with an error naming the nearest valid values, never by silently
rounding — rounding would return a clip of a different length or size than was asked for.

**Frames must be `1 + 8n`** (9, 17, 73, 145, …). The VAE compresses 8x temporally, plus one
independent leading frame.

**Height and width must be divisible by 24.** This is stricter than the 16 the VAE's encoder alone
implies: a `Stem` layer with stride 3 sits in front of it. `1280` is the trap here — divisible by 16
but not by 3 — which is why the reference width is **1248**.

## FAQ

**Can I batch several videos in one request?** You can pass a list of requests, but they are served
one at a time, not fused: `max_num_seqs` is 1 for this pipeline. That is a correctness requirement, not
a tuning default — the DiT shares one rotary position table and one KV cache scope across a batch, so a
fused pair would generate the second video against the first one's positions. If you raise
`max_num_seqs`, the pipeline raises rather than returning quietly wrong videos. Throughput comes from
running several engines, not from batching within one.

**Why is memory flat in clip length?** The KV window keeps at most three chunks resident (the first
chunk as a global sink plus the two most recent), so a 145-frame clip costs the same memory as a
17-frame one — only wall-clock grows.

**Which acceleration features work?** None of the diffusion accelerations apply in this MVP.
Cache-DiT and TeaCache need roughly ten or more steps to find step-to-step redundancy and there are
only two. Tensor, sequence, and CFG parallelism are not wired up; CFG parallelism is structurally
inapplicable, since there is no negative branch to place on a second rank.

**Is streaming supported?** Not yet. The model is natively streaming-capable — that is what the
chunk-autoregressive rollout is for — but this integration covers offline fixed-length editing only.
