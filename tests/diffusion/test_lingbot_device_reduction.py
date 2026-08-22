# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of the LingBot-Video device-reduction fast path.

LingBot-Video decodes with an AutoencoderKLWan VAE and denormalizes as
``clamp(-1, 1)`` then ``(x + 1) / 2`` inside the worker (``_decode_latents``),
which equals ``reduce_video_to_uint8_frames``'s ``(x*0.5+0.5).clamp(0, 1)``.
These tests pin that (1) the post-process forwards a uint8 video without
widening it back to float and (2) the reduced uint8 frames equal LingBot's
float denorm at the encoder input.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import get_lingbot_video_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def test_lingbot_post_process_passes_uint8_video_through() -> None:
    """A reduced uint8 video must not be widened back to float."""
    post_process = get_lingbot_video_post_process_func(SimpleNamespace())
    video = torch.randint(0, 256, (8, 16, 16, 3), dtype=torch.uint8)  # [F, H, W, C]

    result = post_process({"video": video}, sampling_params=OmniDiffusionSamplingParams(output_type="np"))

    assert result["video"].dtype == np.uint8
    np.testing.assert_array_equal(result["video"], video.numpy())


@requires_gpu
def test_lingbot_device_reduction_matches_float_path() -> None:
    """Reduced uint8 frames must equal LingBot's float denorm at encoder input."""
    from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

    torch.manual_seed(0)
    # VAE decode output: [B, C, F, H, W] bf16 slightly outside [-1, 1].
    raw = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    # LingBot's float path (see _decode_latents): clamp(-1,1) -> (x+1)/2 -> [F,H,W,C].
    float_frames = raw.float().clamp(-1, 1)
    float_frames = ((float_frames + 1.0) / 2.0).permute(0, 2, 3, 4, 1).cpu()[0]

    reduced = reduce_video_to_uint8_frames(raw)[0].cpu()
    assert reduced.dtype == torch.uint8

    expected = _coerce_video_to_uint8_frames(float_frames.numpy())
    produced = _coerce_video_to_uint8_frames(reduced.numpy())
    np.testing.assert_array_equal(produced, expected)
