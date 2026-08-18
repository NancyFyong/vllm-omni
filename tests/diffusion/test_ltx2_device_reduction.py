# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence of the LTX-2 device-reduction fast path.

LTX-2 runs ``postprocess_video`` inside the worker runtime, so when
``reduce_video_on_device`` is enabled the runtime instead reduces the decoded
video to uint8 on the GPU and the engine post-process packages it as-is. Two
guards:

1. the engine post-process passes a uint8 video through unchanged (no copy /
   denormalize), so the reduced frames are not corrupted downstream;
2. the reduced uint8 frames equal the float ``postprocess_video`` frames at the
   encoder input (GPU).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.ltx2 import ltx2_components
from vllm_omni.diffusion.models.ltx2.ltx2_components import get_ltx2_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def test_ltx2_post_process_passes_uint8_video_through(monkeypatch) -> None:
    """The engine post-process must forward a reduced uint8 video verbatim."""
    monkeypatch.setattr(ltx2_components, "_detect_vocoder_output_sample_rate", lambda *a, **k: 48000)
    post_process = get_ltx2_post_process_func(SimpleNamespace(model="dummy", revision=None))

    video = torch.randint(0, 256, (1, 4, 16, 16, 3), dtype=torch.uint8)
    audio = torch.zeros(1, 128)
    result = post_process((video, audio))

    assert result["video"] is video  # no copy, no denormalize
    assert result["audio_sample_rate"] == 48000


@requires_gpu
def test_ltx2_device_reduction_matches_float_path() -> None:
    """Reduced uint8 frames must equal float postprocess frames at the encoder."""
    from diffusers.video_processor import VideoProcessor

    from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

    torch.manual_seed(0)
    # Raw LTX VAE decode: [B, C, F, H, W] bf16 slightly outside [-1, 1].
    raw = torch.rand(1, 3, 5, 48, 64, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    # Float path mirrors ltx2_runtime: postprocess_video(raw, "np").
    float_np = VideoProcessor(vae_scale_factor=32).postprocess_video(raw.float(), output_type="np")
    reduced = reduce_video_to_uint8_frames(raw)
    assert reduced.dtype == torch.uint8

    reduced_np = reduced.cpu().numpy()
    for i in range(float_np.shape[0]):
        expected = _coerce_video_to_uint8_frames(float_np[i])
        produced = _coerce_video_to_uint8_frames(reduced_np[i])
        np.testing.assert_array_equal(produced, expected)
