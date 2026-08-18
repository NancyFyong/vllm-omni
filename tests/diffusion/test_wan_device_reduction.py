# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end equivalence of the WAN device-reduction fast path.

When ``reduce_video_on_device`` is enabled, WAN's ``forward`` reduces the
decoded video to uint8 frames on the GPU and its ``post_process_func`` passes
those frames through unchanged. This must produce byte-identical encoder input
to the historical float path (diffusers postprocess + the API server's ``*255``
rounding); otherwise the flag would silently change pixels.

These tests exercise the *real* WAN post-process and the *real* encoder coercion
so a future change to either side that breaks the equivalence turns red.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import get_wan22_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames
from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def _encoder_frames(video_payload: np.ndarray) -> list[np.ndarray]:
    """Run the real encoder coercion on each video in a [B, F, H, W, C] payload."""
    return [_coerce_video_to_uint8_frames(video_payload[i]) for i in range(video_payload.shape[0])]


@requires_gpu
def test_wan_device_reduction_matches_float_path_at_encoder_input() -> None:
    """flag-on frames must equal flag-off frames, byte for byte, at the encoder."""
    torch.manual_seed(0)
    # Synthetic VAE decode output: [B, C, F, H, W] bf16 in ~[-1, 1]. Values are
    # pushed slightly outside [-1, 1] so the clamp is exercised on both paths.
    vae_out = torch.rand(2, 3, 8, 64, 96, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2

    post_process = get_wan22_post_process_func(SimpleNamespace())
    sampling = SimpleNamespace(output_type=None, enable_frame_interpolation=False)

    # Float path (flag off). Production widens bf16->f32 over SHM before the
    # engine postprocess, so widen here too for an honest comparison.
    float_out = post_process(vae_out.float(), output_type="np", sampling_params=sampling)
    float_frames = _encoder_frames(float_out["payload"]["video"])

    # Device-reduced path (flag on): reduce in "forward", pass through in post.
    reduced = reduce_video_to_uint8_frames(vae_out)
    reduced_out = post_process(reduced, output_type="np", sampling_params=sampling)
    assert reduced_out["payload"]["video"].dtype == np.uint8
    reduced_frames = _encoder_frames(reduced_out["payload"]["video"])

    assert len(float_frames) == len(reduced_frames) == 2
    for expected, produced in zip(float_frames, reduced_frames, strict=True):
        assert produced.shape == expected.shape == (8, 64, 96, 3)
        np.testing.assert_array_equal(produced, expected)


@requires_gpu
def test_wan_post_process_passthrough_keeps_uint8_without_rescaling() -> None:
    """The uint8 branch must not denormalize/scale already-[0,255] frames."""
    frames = torch.randint(0, 256, (1, 4, 16, 16, 3), dtype=torch.uint8, device="cuda:0")

    out = get_wan22_post_process_func(SimpleNamespace())(
        frames, output_type="np", sampling_params=SimpleNamespace(output_type=None)
    )

    np.testing.assert_array_equal(out["payload"]["video"], frames.cpu().numpy())


@requires_gpu
def test_wan_post_process_still_denormalizes_float_input() -> None:
    """A float tensor (flag off / ineligible request) must take the diffusers path."""
    video = torch.rand(1, 3, 4, 16, 16, device="cuda:0", dtype=torch.float32) * 2 - 1

    out = get_wan22_post_process_func(SimpleNamespace())(
        video, output_type="np", sampling_params=SimpleNamespace(output_type=None)
    )

    payload = out["payload"]["video"]
    # diffusers postprocess_video(output_type="np") returns float32 in [0, 1].
    assert payload.dtype == np.float32
    assert float(payload.min()) >= 0.0 and float(payload.max()) <= 1.0
