# Skywork UniPic-3

Skywork's UniPic-3 family fine-tunes `Qwen/Qwen-Image-Edit` — the transformer
config (60 layers × 24 heads, `axes_dims_rope=[16, 56, 56]`, `in_channels=64`)
and the surrounding stack (VAE `AutoencoderKLQwenImage`, text encoder
`Qwen2_5_VLForConditionalGeneration`, scheduler `FlowMatchEulerDiscreteScheduler`)
match Qwen-Image-Edit exactly. All three checkpoints declare
`_class_name: QwenImageEditPipeline` in `model_index.json`, so vllm-omni's
existing `QwenImageEditPipeline` loads them without any adapter — point
`--model` at a local snapshot and use `examples/offline_inference/image_to_image/image_edit.py`.

| Checkpoint | HF repo | Sampling | Notes |
| :--- | :--- | :--- | :--- |
| Teacher | `Skywork/Unipic3` | 50-step, `cfg_scale=4.0` | Highest fidelity, slowest. |
| DMD (distilled) | `Skywork/Unipic3-DMD` | 4–8 step, `cfg_scale=1.0` | Distribution-matched, fast inference. Use `ema_transformer/`. |
| Consistency (distilled) | `Skywork/Unipic3-Consistency-Model` | few-step | Ships only `transformer/` + `ema_transformer/`; assemble before use. |

## 1. Download

```bash
hf download Skywork/Unipic3 --local-dir /path/to/Unipic3
hf download Skywork/Unipic3-DMD --local-dir /path/to/Unipic3-DMD
hf download Skywork/Unipic3-Consistency-Model --local-dir /path/to/Unipic3-CM
```

## 2. Assemble the DMD and Consistency-Model variants

Both distilled checkpoints ship an `ema_transformer/` folder — the README on
each HF repo recommends inference from the EMA weights. The DMD checkpoint
also ships a non-EMA `transformer/`. The Consistency-Model checkpoint ships
**no** VAE / text_encoder / scheduler / tokenizer / processor, so it must be
composed with the teacher's non-transformer subfolders before it can be loaded
by vllm-omni.

`prepare_unipic3.py` handles both cases with symlinks (no extra disk):

```bash
# Assemble DMD with EMA transformer weights.
python prepare_unipic3.py \
    --variant dmd \
    --source /path/to/Unipic3-DMD \
    --output /path/to/Unipic3-DMD-Ready

# Assemble Consistency-Model on top of the teacher's non-transformer folders.
python prepare_unipic3.py \
    --variant cm \
    --source /path/to/Unipic3-CM \
    --teacher /path/to/Unipic3 \
    --output /path/to/Unipic3-CM-Ready
```

`--use-ema` is on by default (recommended by the model cards); pass
`--no-use-ema` to use the non-EMA `transformer/` weights on the DMD variant.

## 3. Run

Reuse the image_to_image CLI. The examples below assume you've already
downloaded a sample image (see `../image_to_image/image_to_image.md`):

```bash
# Teacher (high-fidelity, 50 steps).
python ../image_to_image/image_edit.py \
    --model /path/to/Unipic3 \
    --image qwen-bear.png \
    --prompt "Turn this bear into an astronaut floating above a starry ocean." \
    --output unipic3_teacher.png \
    --num-inference-steps 50 \
    --cfg-scale 4.0

# DMD few-step inference.
python ../image_to_image/image_edit.py \
    --model /path/to/Unipic3-DMD-Ready \
    --image qwen-bear.png \
    --prompt "Turn this bear into an astronaut floating above a starry ocean." \
    --output unipic3_dmd.png \
    --num-inference-steps 8 \
    --cfg-scale 1.0

# Consistency-Model few-step inference.
python ../image_to_image/image_edit.py \
    --model /path/to/Unipic3-CM-Ready \
    --image qwen-bear.png \
    --prompt "Turn this bear into an astronaut floating above a starry ocean." \
    --output unipic3_cm.png \
    --num-inference-steps 8 \
    --cfg-scale 1.0
```

Key knobs:

- `--num-inference-steps`: 50 for the teacher; 4–8 for DMD / Consistency-Model.
- `--cfg-scale`: 4.0 for the teacher (with `--negative-prompt`); 1.0 for the
  distilled variants (they are trained without CFG).
- Multi-image composition (2–6 images): pass multiple paths after `--image`.
- Cache-DiT / TP / USP / CFG-Parallel: all inherited from `QwenImageEditPipeline`
  (see `../image_to_image/image_to_image.md`).
