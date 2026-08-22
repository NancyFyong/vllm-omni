# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end equivalence of the HunyuanVideo-1.5 device-reduction fast path.

When ``enable_device_postprocess`` is enabled, HunyuanVideo-1.5's ``forward``
reduces the decoded video to uint8 frames on the GPU and its
``post_process_func`` passes those frames through. The engine calls the
post-process without an ``output_type`` (so it defaults to ``"pil"``), hence the
uint8 branch must produce PIL frames byte-identical to the float path
(``Image.fromarray`` on uint8 == diffusers ``numpy_to_pil``). The ``"np"`` path
is also covered for completeness.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    get_hunyuan_video_15_post_process_func,
)
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


@requires_gpu
def test_hunyuan_device_reduction_matches_float_path_pil() -> None:
    """Default (pil) path: uint8->PIL frames must equal float->PIL frames."""
    torch.manual_seed(0)
    # Synthetic VAE decode output: [B, C, F, H, W] bf16 slightly outside [-1, 1].
    vae_out = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    post_process = get_hunyuan_video_15_post_process_func(SimpleNamespace())

    # Float path: production widens bf16->f32 over SHM before the engine post.
    float_frames = post_process(vae_out.float(), output_type="pil")

    # Device-reduced path: reduce in "forward", pass through (as PIL) in post.
    reduced = reduce_video_to_uint8_frames(vae_out)
    reduced_frames = post_process(reduced, output_type="pil")

    assert len(float_frames) == len(reduced_frames) == 5
    for expected, produced in zip(float_frames, reduced_frames, strict=True):
        np.testing.assert_array_equal(np.asarray(produced), np.asarray(expected))


@requires_gpu
def test_hunyuan_device_reduction_matches_float_path_np() -> None:
    """np path: uint8 frames must equal float frames after the encoder coercion."""
    from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

    torch.manual_seed(1)
    vae_out = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    post_process = get_hunyuan_video_15_post_process_func(SimpleNamespace())

    # The device path computes in float32, so a float32 reference must match exactly.
    # The engine denormalizes the bf16 decode in its own dtype, which lands on a
    # different 255th for some pixels; that path is only bounded, not matched.
    widened_np = post_process(vae_out.float(), output_type="np")  # [B, F, H, W, C] float32
    native_np = post_process(vae_out, output_type="np")
    reduced = reduce_video_to_uint8_frames(vae_out)
    reduced_np = post_process(reduced, output_type="np")  # [B, F, H, W, C] uint8
    assert reduced_np.dtype == np.uint8

    for i in range(widened_np.shape[0]):
        expected = _coerce_video_to_uint8_frames(widened_np[i])
        produced = _coerce_video_to_uint8_frames(reduced_np[i])
        np.testing.assert_array_equal(produced, expected)
        native = _coerce_video_to_uint8_frames(native_np[i])
        assert np.abs(produced.astype(np.int16) - native.astype(np.int16)).max() <= 1


@requires_gpu
def test_hunyuan_post_process_still_denormalizes_float_input() -> None:
    """A float tensor (flag off) must take the diffusers path and yield PIL."""
    from PIL import Image

    video = torch.rand(1, 3, 3, 32, 32, device="cuda:0", dtype=torch.float32) * 2 - 1

    frames = get_hunyuan_video_15_post_process_func(SimpleNamespace())(video, output_type="pil")

    assert len(frames) == 3
    assert all(isinstance(f, Image.Image) for f in frames)
