# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side reduction of decoded video tensors to uint8 frames.

Video pipelines emit a decoded tensor in the VAE's native form -- floating
point (often bf16), channel-first ``[B, C, F, H, W]`` layout, values in the
normalized ``[-1, 1]`` range. The serving path only ever ships uint8 RGB
frames to the encoder, so that float tensor is 4x larger than the bytes that
actually leave the process.

Historically the float->uint8 reduction happened *after* the device-to-host
copy (diffusers postprocess in the engine, then ``*255`` in the API server), so
every hop between the GPU and the encoder moved the oversized float payload.

This module performs the exact same arithmetic on the GPU *before* the D2H
copy, so the copy -- and every downstream hop -- carries uint8 frames instead.
The output matches diffusers' ``VideoProcessor.postprocess_video(output_type=
"np")`` followed by the API server's ``*255`` rounding, bit for bit.
"""

from __future__ import annotations

import torch

# diffusers VaeImageProcessor.denormalize: (x * 0.5 + 0.5).clamp(0, 1)
_DENORM_SCALE = 0.5
_DENORM_SHIFT = 0.5


def reduce_video_to_uint8_frames(video: torch.Tensor, *, do_denormalize: bool = True) -> torch.Tensor:
    """Reduce a decoded video tensor to uint8 RGB frames on its current device.

    Args:
        video: Decoded video tensor shaped ``[B, C, F, H, W]`` (the VAE output
            layout used by diffusers video pipelines), any float dtype, on any
            device. ``C`` is the colour channel count (3 for RGB).
        do_denormalize: When ``True`` (diffusers default) map ``[-1, 1]`` to
            ``[0, 1]`` before scaling. Set ``False`` for pipelines whose
            processor has ``do_normalize=False`` and already emit ``[0, 1]``.

    Returns:
        A ``uint8`` tensor shaped ``[B, F, H, W, C]`` on the same device. This
        is the frame layout the MP4 encoder consumes; no host copy is made
        here, so the caller controls when the (now 4x smaller) D2H happens.

    The arithmetic reproduces, in order:
      * ``VaeImageProcessor.denormalize``: ``(x * 0.5 + 0.5).clamp(0, 1)``
      * ``VideoProcessor.postprocess_video`` layout: per batch item
        ``permute(1, 0, 2, 3)`` (``[C,F,H,W]`` -> ``[F,C,H,W]``) then
        ``pt_to_numpy`` ``permute(0, 2, 3, 1)`` (-> ``[F,H,W,C]``)
      * the API server's ``(x * 255).round().astype(uint8)``
    """
    if video.dim() != 5:
        raise ValueError(f"expected a [B, C, F, H, W] video tensor, got shape {tuple(video.shape)}")

    # Do the elementwise math in float32 so rounding matches the numpy path,
    # which promotes to float before scaling.
    frames = video.to(torch.float32)
    if do_denormalize:
        frames = frames.mul(_DENORM_SCALE).add(_DENORM_SHIFT).clamp_(0.0, 1.0)
    else:
        frames = frames.clamp(0.0, 1.0)

    # [B, C, F, H, W] -> [B, F, H, W, C] to match the encoder's frame layout.
    frames = frames.permute(0, 2, 3, 4, 1)

    # (x * 255).round() -> uint8, matching numpy_to_pil / _coerce_video_to_uint8.
    frames = frames.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
    return frames.contiguous()
