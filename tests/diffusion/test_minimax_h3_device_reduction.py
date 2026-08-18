# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of the MiniMax-H3 device-reduction fast path.

MiniMax-H3's decode output is already in ``[0, 1]`` (the post-process clamps
without denormalizing), so the device reduction runs with
``do_denormalize=False``. The engine calls ``_minimax_h3_post_process`` without
an ``output_type`` (defaults to ``"np"``), so the uint8 branch must produce the
same encoder frames the float path would, and audio must be untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_post_process
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def test_minimax_h3_post_process_passes_uint8_video_through() -> None:
    """A reduced uint8 video must survive post-process without re-scaling."""
    video = torch.randint(0, 256, (1, 4, 8, 8, 3), dtype=torch.uint8)  # [B, F, H, W, C]
    audio = torch.zeros(1, 100)

    result = _minimax_h3_post_process((video, audio), output_type="np")

    assert isinstance(result["video"], list) and len(result["video"]) == 1
    np.testing.assert_array_equal(result["video"][0], video[0].numpy())


@requires_gpu
def test_minimax_h3_device_reduction_matches_float_path() -> None:
    """Reduced uint8 frames must equal the float path at the encoder input."""
    from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

    torch.manual_seed(0)
    # MiniMax-H3 decode output: [B, C, F, H, W] in ~[0, 1] (pushed out of range
    # to exercise the clamp on both paths).
    video = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.float16) * 1.2 - 0.1
    audio = torch.zeros(1, 100, device="cuda:0")

    float_out = _minimax_h3_post_process((video, audio), output_type="np")
    reduced = reduce_video_to_uint8_frames(video, do_denormalize=False)
    assert reduced.dtype == torch.uint8
    reduced_out = _minimax_h3_post_process((reduced, audio), output_type="np")
    assert reduced_out["video"][0].dtype == np.uint8

    for float_sample, uint8_sample in zip(float_out["video"], reduced_out["video"], strict=True):
        expected = _coerce_video_to_uint8_frames(float_sample)
        produced = _coerce_video_to_uint8_frames(uint8_sample)
        np.testing.assert_array_equal(produced, expected)
