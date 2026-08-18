# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU correctness/size tests for device-side video reduction.

These guard the root-cause optimization from RFC #6212: reducing a decoded
video tensor to uint8 frames *on the GPU before the D2H copy*. The reduction
must be byte-identical to the current production output (SHM widens bf16->f32,
then diffusers postprocess, then the API server's ``*255`` rounding), otherwise
enabling it would silently change pixels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess.device_reduction import reduce_video_to_uint8_frames

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def _device() -> str:
    return "cuda:0"


def _reference_uint8(video: torch.Tensor) -> np.ndarray:
    """The exact current production output for the same input.

    Production widens bf16->f32 during SHM transport (ipc._tensor_to_shm), so
    the engine's diffusers postprocess runs on float32; the API server then
    rounds to uint8. Reproduce that ordering so the comparison is honest.
    """
    from diffusers.video_processor import VideoProcessor

    widened = video.float()  # SHM transport widening happens before postprocess
    processed = VideoProcessor(vae_scale_factor=8).postprocess_video(widened, output_type="np")
    scaled = np.clip(processed.astype(np.float32), 0.0, 1.0) * 255.0
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint8)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_device_reduction_is_byte_identical_to_production(dtype: torch.dtype) -> None:
    """Enabling the fast path must not move a single pixel value."""
    torch.manual_seed(0)
    # Values slightly outside [-1, 1] so the clamp is actually exercised.
    video = (torch.rand(1, 3, 8, 64, 96, device=_device()) * 2.4 - 1.2).to(dtype)

    reference = _reference_uint8(video)
    produced = reduce_video_to_uint8_frames(video).cpu().numpy()

    assert produced.shape == reference.shape == (1, 8, 64, 96, 3)
    assert produced.dtype == np.uint8
    np.testing.assert_array_equal(produced, reference)


@requires_gpu
def test_device_reduction_shrinks_hop1_payload_4x() -> None:
    """The whole point: the D2H payload drops from float32 to uint8 (4x)."""
    video = torch.rand(1, 3, 8, 704, 1280, device=_device(), dtype=torch.bfloat16) * 2 - 1

    # Production transports bf16 widened to float32 over SHM.
    widened_f32_bytes = video.numel() * 4
    reduced = reduce_video_to_uint8_frames(video)
    reduced_bytes = reduced.numel() * reduced.element_size()

    assert reduced.dtype == torch.uint8
    assert widened_f32_bytes / reduced_bytes == pytest.approx(4.0, rel=1e-6)


@requires_gpu
def test_device_reduction_stays_on_device() -> None:
    """Reduction must happen before D2H, so the result is still on the GPU."""
    video = torch.rand(1, 3, 4, 32, 48, device=_device(), dtype=torch.bfloat16) * 2 - 1
    reduced = reduce_video_to_uint8_frames(video)
    assert reduced.device.type == "cuda"
    assert reduced.is_contiguous()


@requires_gpu
def test_device_reduction_without_denormalize_matches_already_unit_range() -> None:
    """Pipelines with do_normalize=False emit [0,1]; skip the denorm step."""
    video = torch.rand(1, 3, 4, 16, 16, device=_device(), dtype=torch.float32)  # already [0,1]

    produced = reduce_video_to_uint8_frames(video, do_denormalize=False).cpu().numpy()

    expected = video.permute(0, 2, 3, 4, 1).cpu().numpy()
    expected = np.rint(np.clip(expected, 0, 1) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(produced, expected)


def test_device_reduction_rejects_non_5d() -> None:
    """A wrong-rank tensor is a programming error, not a silent reshape."""
    with pytest.raises(ValueError, match=r"\[B, C, F, H, W\]"):
        reduce_video_to_uint8_frames(torch.rand(3, 8, 64, 64))
