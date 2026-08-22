# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side reduction of decoded video tensors to uint8 frames."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

# VaeImageProcessor.denormalize: (x * 0.5 + 0.5).clamp(0, 1)
_DENORM_SCALE = 0.5
_DENORM_SHIFT = 0.5

# Distinguishes "this pipeline has no output type to check" from output_type=None,
# which is a real value that must keep the float path.
_UNSET = object()


def _as_sampling_params_list(sampling_params: Any) -> list[Any]:
    """Normalize one sampling params object, a batch of them, or nothing."""
    if sampling_params is None:
        return []
    if isinstance(sampling_params, Iterable) and not isinstance(sampling_params, str | bytes):
        return list(sampling_params)
    return [sampling_params]


def should_enable_device_postprocess(
    od_config: Any,
    sampling_params: Any = None,
    *,
    output_type: Any = _UNSET,
    blocked: bool = False,
) -> bool:
    """Whether a decoded video may be reduced to uint8 before the D2H copy.

    Shared by every video pipeline and deliberately conservative: anything
    unrecognised keeps the float path. ``sampling_params`` may be one request or
    the whole batch, since a single request needing float closes the gate.

    Args:
        output_type: Omit for pipelines that postprocess to a fixed format
            regardless of the request; passing ``None`` keeps the float path.
        blocked: Model-specific veto, evaluated by the caller (Cosmos3 guardrails
            must see the float video, LingBot text-to-image has no video path).
    """
    if blocked:
        return False

    transport = getattr(od_config, "video_output_transport", None)
    if transport is None or not getattr(transport, "enable_device_postprocess", False):
        return False

    if output_type is not _UNSET and output_type != "np":
        return False

    for params in _as_sampling_params_list(sampling_params):
        if (getattr(params, "output_type", None) or "np") != "np":
            return False
        # Frame interpolation consumes the float [B, C, F, H, W] video.
        if getattr(params, "enable_frame_interpolation", False):
            return False

    return True


def is_device_reduced(video: Any) -> bool:
    """Whether ``video`` is already uint8 frames from the device-side reduction.

    Used by each ``post_process`` to skip the float denormalize path.
    """
    return isinstance(video, torch.Tensor) and video.dtype == torch.uint8


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
