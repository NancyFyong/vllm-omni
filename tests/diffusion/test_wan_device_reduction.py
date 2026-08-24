# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WAN reference coverage for the typed pre-D2H media contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, VideoOutputTransportConfig
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm, unpack_diffusion_output_shm
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import get_wan22_post_process_func
from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.entrypoints.openai.video_api_utils import _coerce_video_to_uint8_frames
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_GPU = torch.accelerator.is_available() if hasattr(torch, "accelerator") else torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not _GPU, reason="device reduction is a GPU path")


def _decoded_wan_video(tensor: torch.Tensor) -> DiffusionMediaOutput:
    return DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=tensor,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BCTHW,
                encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
            ),
        )
    )


def _config(*, enabled: bool) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model=None,
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=enabled),
    )


def _encoder_frames(video_payload: np.ndarray) -> list[np.ndarray]:
    return [_coerce_video_to_uint8_frames(video_payload[i]) for i in range(video_payload.shape[0])]


@requires_gpu
def test_wan_media_reduction_matches_float32_reference_and_bounds_native_path() -> None:
    torch.manual_seed(0)
    vae_out = torch.rand(2, 3, 8, 64, 96, device="cuda:0", dtype=torch.bfloat16) * 2.4 - 1.2
    sampling = OmniDiffusionSamplingParams(output_type=None, enable_frame_interpolation=False)

    legacy_postprocess = get_wan22_post_process_func(_config(enabled=False))
    widened_out = legacy_postprocess(vae_out.float(), output_type="np", sampling_params=sampling)
    widened_frames = _encoder_frames(widened_out["payload"]["video"])
    native_out = legacy_postprocess(vae_out, output_type="np", sampling_params=sampling)
    native_frames = _encoder_frames(native_out["payload"]["video"])

    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(vae_out),
        od_config=_config(enabled=True),
        sampling_params=sampling,
    )
    reduced_out = finalize_diffusion_media(prepared, sampling_params=sampling)
    assert prepared.video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES
    assert reduced_out["payload"]["video"].dtype == np.uint8
    reduced_frames = _encoder_frames(reduced_out["payload"]["video"])

    assert len(widened_frames) == len(reduced_frames) == 2
    for expected, native, produced in zip(widened_frames, native_frames, reduced_frames, strict=True):
        assert produced.shape == expected.shape == (8, 64, 96, 3)
        np.testing.assert_array_equal(produced, expected)
        deviation = np.abs(produced.astype(np.int16) - native.astype(np.int16))
        assert deviation.max() <= 1


@requires_gpu
def test_wan_uint8_media_finalization_does_not_rescale() -> None:
    frames = torch.randint(0, 256, (1, 4, 16, 16, 3), dtype=torch.uint8, device="cuda:0")
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=frames,
            spec=VideoTensorSpec(
                layout=VideoTensorLayout.BTHWC,
                encoding=VideoTensorEncoding.UINT8_FRAMES,
                value_range=VideoValueRange.ZERO_TO_255,
            ),
        ),
        prepared_for_transport=True,
    )

    out = finalize_diffusion_media(media, sampling_params=OmniDiffusionSamplingParams(output_type="np"))

    np.testing.assert_array_equal(out["payload"]["video"], frames.cpu().numpy())


@requires_gpu
def test_diffusion_output_to_cpu_moves_wan_media_off_device() -> None:
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(torch.randn(1, 3, 2, 4, 5, device="cuda:0")),
        od_config=_config(enabled=False),
        sampling_params=OmniDiffusionSamplingParams(output_type="np"),
    )
    output = DiffusionOutput(media=prepared, to_cpu=True)

    assert output.media is not None
    assert output.media.video.tensor.device.type == "cpu"


@requires_gpu
def test_small_typed_media_is_on_cpu_before_ipc() -> None:
    sampling = OmniDiffusionSamplingParams(output_type="np")
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(torch.randn(1, 3, 2, 4, 5, device="cuda:0")),
        od_config=_config(enabled=True),
        sampling_params=sampling,
    )
    output = DiffusionOutput(media=prepared)

    pack_diffusion_output_shm(output)

    assert isinstance(output.media, dict)
    packed_tensor = output.media["video"]["tensor"]
    assert isinstance(packed_tensor, torch.Tensor)
    assert packed_tensor.device.type == "cpu"
    unpack_diffusion_output_shm(output)


@requires_gpu
def test_wan_float_media_finalization_matches_the_legacy_path() -> None:
    video = torch.rand(1, 3, 4, 16, 16, device="cuda:0", dtype=torch.bfloat16) * 2 - 1
    sampling = OmniDiffusionSamplingParams(output_type="np")
    prepared = prepare_diffusion_media_for_transport(
        _decoded_wan_video(video),
        od_config=_config(enabled=False),
        sampling_params=sampling,
    )

    produced = finalize_diffusion_media(prepared, sampling_params=sampling)
    expected = get_wan22_post_process_func(_config(enabled=False))(
        video,
        output_type="np",
        sampling_params=sampling,
    )

    assert prepared.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    np.testing.assert_array_equal(produced["payload"]["video"], expected["payload"]["video"])
