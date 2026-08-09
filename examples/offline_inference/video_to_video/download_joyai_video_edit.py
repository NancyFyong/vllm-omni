# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download the JoyAI-Video-Edit weights and synthesize the pipeline's ``model_index.json``.

JoyAI-Video-Edit ships as a raw checkpoint tree rather than a diffusers pipeline, so there is no
``model_index.json`` upstream. vllm-omni dispatches on the ``_class_name`` key in that file, so this
script writes one whose value matches the registry entry for ``JoyAIVideoEditPipeline``.

Layout produced under ``--output-dir``::

    joyai_video_edit/
    |-- model_index.json                                    (synthesized here)
    |-- dit/joyai_video_edit_dit_0804.pth                   30.3 GiB
    |-- vae/{config.json,diffusion_pytorch_model.safetensors} 1.4 GiB
    `-- MiMo-VL-7B-RL-2508/...                              15.5 GiB

Total ~47 GiB. Usage::

    python download_joyai_video_edit.py --output-dir /path/to/models/joyai_video_edit
"""

import argparse
import json
import os
import time

from huggingface_hub import snapshot_download

DIT_REPO = "jdopensource/JoyAI-Video-Edit"
CONDITION_ENCODER_REPO = "XiaomiMiMo/MiMo-VL-7B-RL-2508"

# The MLLM condition encoder. The repo also carries README artwork (*.jpeg/*.png/*.gif) that we skip.
CONDITION_ENCODER_PATTERNS = ["*.safetensors", "*.json", "*.txt"]

# Relative to the DiT repo root. The two ONNX detectors named in upstream's DEPLOYMENT.md
# (face_detection_yunet, yolov8n) drive the WebUI's subject picker and are not on the denoise path.
DIT_PATTERNS = ["dit/*", "vae/*"]

DIT_CHECKPOINT = "dit/joyai_video_edit_dit_0804.pth"


def timed_download(repo_id: str, local_dir: str, allow_patterns: list[str] | None = None, retries: int = 3) -> None:
    """Download an HF repo subset, logging elapsed time. Resumes partial downloads."""
    print(f"==> {repo_id} -> {local_dir}", flush=True)
    start = time.time()
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
            break
        except Exception as exc:  # noqa: BLE001 - retry any transport-level failure
            if attempt == retries:
                raise
            print(f"    attempt {attempt}/{retries} failed ({type(exc).__name__}: {exc}); retrying", flush=True)
            time.sleep(30)
    print(f"==> {repo_id} done in {time.time() - start:.1f}s", flush=True)


def write_model_index(output_dir: str) -> None:
    """Write the ``model_index.json`` that vllm-omni's diffusion registry dispatches on.

    ``_class_name`` must match the ``_DIFFUSION_MODELS`` key in vllm_omni/diffusion/registry.py.
    Every other key is surfaced to the pipeline as ``od_config.model_config``.
    """
    config = {
        "_class_name": "JoyAIVideoEditPipeline",
        "_diffusers_version": "0.38.0",
        # Component locations, relative to this file.
        "dit_checkpoint": DIT_CHECKPOINT,
        "vae_path": "vae",
        "condition_encoder_path": "MiMo-VL-7B-RL-2508",
        # Sampling schedule. The model is AR-DMD distilled, so 2 steps is the trained
        # operating point rather than a speed/quality knob, and there is no CFG.
        "num_inference_steps": 2,
        "scheduler_shift": 5.159,
        # Chunked autoregressive rollout: 1 latent frame == 8 pixel frames.
        "chunk_size": 1,
        # KV window: a global sink chunk plus the 2 most recent chunks.
        "local_window_size": 3,
        "global_sink_chunk": True,
        "kv_cache_pre_rope": True,
        # Default resolution. Width must stay a multiple of 3 for the VAE Stem (see the VAE module).
        "default_height": 720,
        "default_width": 1248,
        "max_condition_tokens": 1024,
    }
    path = os.path.join(output_dir, "model_index.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"==> wrote {path}", flush=True)


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timed_download(DIT_REPO, output_dir, allow_patterns=DIT_PATTERNS)
    timed_download(
        CONDITION_ENCODER_REPO,
        os.path.join(output_dir, "MiMo-VL-7B-RL-2508"),
        allow_patterns=CONDITION_ENCODER_PATTERNS,
    )
    write_model_index(output_dir)

    checkpoint = os.path.join(output_dir, DIT_CHECKPOINT)
    if not os.path.exists(checkpoint):
        raise RuntimeError(f"DiT checkpoint missing after download: {checkpoint}")
    print(f"==> all weights present under {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./joyai_video_edit",
        help="Directory to download the weights into",
    )
    main(parser.parse_args().output_dir)
