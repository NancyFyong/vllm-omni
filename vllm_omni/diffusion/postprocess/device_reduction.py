# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side reduction of decoded video tensors to uint8 frames."""

from __future__ import annotations

import torch

# VaeImageProcessor.denormalize: (x * 0.5 + 0.5).clamp(0, 1)
_DENORM_SCALE = 0.5
_DENORM_SHIFT = 0.5


def reduce_video_to_uint8_frames(video: torch.Tensor, *, do_denormalize: bool = True) -> torch.Tensor:
    """Reduce a decoded ``[B, C, F, H, W]`` video to uint8 ``[B, F, H, W, C]`` frames.

    Runs denormalize/clamp/permute/round on the input's device so the following
    D2H copy carries uint8 instead of float. The result matches
    ``VideoProcessor.postprocess_video(output_type="np")`` then the ``*255``
    rounding done in the API server. Pass ``do_denormalize=False`` for VAEs that
    already emit ``[0, 1]``.
    """
    if video.dim() != 5:
        raise ValueError(f"expected a [B, C, F, H, W] video tensor, got shape {tuple(video.shape)}")

    # Match the numpy path, which promotes to float before scaling.
    frames = video.to(torch.float32)
    if do_denormalize:
        frames = frames.mul(_DENORM_SCALE).add(_DENORM_SHIFT).clamp_(0.0, 1.0)
    else:
        frames = frames.clamp(0.0, 1.0)
    frames = frames.permute(0, 2, 3, 4, 1)
    frames = frames.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
    return frames.contiguous()
