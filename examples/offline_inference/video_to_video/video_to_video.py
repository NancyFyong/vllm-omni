# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline instruction-driven video editing with JoyAI-Video-Edit.

Download the weights first::

    python download_joyai_video_edit.py --output-dir /path/to/models/joyai_video_edit

Then edit a clip::

    python video_to_video.py \\
        --model /path/to/models/joyai_video_edit \\
        --video input.mp4 \\
        --prompt "make it snow" \\
        --num-frames 73

The frame count must be ``1 + 8n`` (the VAE compresses 8x temporally) and the resolution divisible by
24 -- the pipeline raises with the nearest valid values rather than silently rounding, since either
would return a clip of the wrong length or size.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Local weights directory (must be a path, not a HF repo id).")
    parser.add_argument("--video", required=True, help="Source video path or URL to edit.")
    parser.add_argument("--prompt", required=True, help="Edit instruction, e.g. 'make it snow'.")
    parser.add_argument("--output", default="joyai_video_edit_output.mp4")
    parser.add_argument("--height", type=int, default=720, help="Must be divisible by 24.")
    parser.add_argument("--width", type=int, default=1248, help="Must be divisible by 24; 1280 is invalid.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Must be 1 + 8n. Defaults to the longest valid prefix of the source clip.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Defaults to the source video's frame rate.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_frames(output: Any) -> np.ndarray:
    """Pull the edited clip out of an ``OmniRequestOutput`` as float frames in ``[0, 1]``.

    Video payloads travel on ``.images`` -- the formatter keys the response off the primary payload
    type, and ``video`` reports as an image response.
    """
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, OmniRequestOutput):
        if not output.images:
            raise ValueError("No frames in OmniRequestOutput.")
        output = output.images[0] if len(output.images) == 1 else output.images

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(output, list):
        output = np.stack([np.asarray(frame) for frame in output], axis=0)

    frames = np.asarray(output)
    if frames.ndim == 5:
        frames = frames[0]
    if np.issubdtype(frames.dtype, np.integer):
        return frames.astype(np.float32) / 255.0
    return frames.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()

    omni = Omni(model=args.model, model_class_name="JoyAIVideoEditPipeline")
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames or 0,
        # Fixed by the AR-DMD distillation; see the deploy YAML.
        num_inference_steps=2,
        seed=args.seed,
        fps=args.fps,
        output_type="np",
    )
    prompt = {"prompt": args.prompt, "multi_modal_data": {"video": args.video}}

    start = time.perf_counter()
    output = omni.generate(prompt, sampling_params)
    elapsed = time.perf_counter() - start

    frames = extract_frames(output)
    print(f"Edited {frames.shape[0]} frames at {frames.shape[2]}x{frames.shape[1]} in {elapsed:.1f}s")

    result = output[0] if isinstance(output, list) and output else output
    # The formatter lifts `metadata["video"]["fps"]` to a top-level `fps` on the multimodal output.
    fps = args.fps or (getattr(result, "multimodal_output", None) or {}).get("fps") or 24

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    export_to_video(list(frames), str(output_path), fps=int(fps))
    print(f"Saved to {output_path}")
    omni.close()


if __name__ == "__main__":
    main()
