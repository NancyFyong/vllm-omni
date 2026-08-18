# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of the Cosmos3 device-reduction fast path.

Cosmos3 runs a guardrail safety check on the *float* video inside the engine
post-process, so the worker only reduces to uint8 when guardrails are disabled.
These tests pin that, with guardrails off, the reduced uint8 video (1) passes
through post-process without the safety check / denormalize and (2) equals the
float ``postprocess_video`` frames at the encoder input.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")

# Server-level guardrail gate off -> is_guardrails_enabled() is False, matching
# the only state in which the worker reduces the video on device.
_OD_CONFIG = SimpleNamespace(model_config={"guardrails": False})
_SAMPLING = SimpleNamespace(extra_args={}, output_type="np")


def test_cosmos3_post_process_passes_uint8_video_through() -> None:
    """With guardrails off, a reduced uint8 video is forwarded verbatim."""
    post_process = get_cosmos3_post_process_func(_OD_CONFIG)
    video = torch.randint(0, 256, (1, 4, 16, 16, 3), dtype=torch.uint8)  # [B, F, H, W, C]

    result = post_process({"video": video}, output_type="np", sampling_params=_SAMPLING)

    # video-only, no audio/action/envelope -> post_process returns the array.
    np.testing.assert_array_equal(result, video.detach().cpu().numpy())


@requires_gpu
def test_cosmos3_device_reduction_matches_float_path() -> None:
    """Reduced uint8 frames must equal the float path at the encoder input."""
    from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

    post_process = get_cosmos3_post_process_func(_OD_CONFIG)
    torch.manual_seed(0)
    # Cosmos3 VAE decode: [B, C, F, H, W] bf16 slightly outside [-1, 1].
    raw = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    float_np = post_process({"video": raw.float()}, output_type="np", sampling_params=_SAMPLING)
    reduced = reduce_video_to_uint8_frames(raw)
    assert reduced.dtype == torch.uint8
    reduced_np = post_process({"video": reduced}, output_type="np", sampling_params=_SAMPLING)
    assert reduced_np.dtype == np.uint8

    for i in range(float_np.shape[0]):
        expected = _coerce_video_to_uint8_frames(float_np[i])
        produced = _coerce_video_to_uint8_frames(reduced_np[i])
        np.testing.assert_array_equal(produced, expected)
