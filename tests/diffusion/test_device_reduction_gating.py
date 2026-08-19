# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates that decide when a video may be reduced to uint8 before the D2H copy.

The reduction is only safe for requests whose downstream path can consume uint8
frames. These gates are the safety mechanism for the whole feature, and they are
enforced in ``forward`` while the uint8 fast paths in ``post_process`` rely on
them, so each one is pinned here plus the fail-closed checks that turn a broken
gate into a loud error instead of a silently wrong video.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    _should_reduce_video_on_device,
    get_wan22_post_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _od_config(*, reduce: bool) -> SimpleNamespace:
    return SimpleNamespace(video_output_transport=VideoOutputTransportConfig(reduce_video_on_device=reduce))


def _sp(output_type: str | None = None, *, interpolate: bool = False) -> SimpleNamespace:
    return SimpleNamespace(output_type=output_type, enable_frame_interpolation=interpolate)


# --- WAN batch-wide gate ---------------------------------------------------


def test_gate_open_when_flag_on_and_whole_batch_wants_np() -> None:
    assert _should_reduce_video_on_device(_od_config(reduce=True), [_sp("np"), _sp(None)], output_type="np") is True


def test_gate_closed_when_flag_off() -> None:
    """Default off must mean no behaviour change at all."""
    assert _should_reduce_video_on_device(_od_config(reduce=False), [_sp("np")], output_type="np") is False


def test_gate_closed_when_config_has_no_transport_section() -> None:
    """A stub/legacy config without the field must not crash or reduce."""
    assert _should_reduce_video_on_device(SimpleNamespace(), [_sp("np")], output_type="np") is False


def test_gate_closed_when_pipeline_output_type_is_not_np() -> None:
    assert _should_reduce_video_on_device(_od_config(reduce=True), [_sp("np")], output_type="pil") is False


def test_gate_closed_when_any_request_wants_a_non_np_output() -> None:
    """One 'latent' request keeps the whole batch on the float path."""
    batch = [_sp("np"), _sp("latent")]
    assert _should_reduce_video_on_device(_od_config(reduce=True), batch, output_type="np") is False


def test_gate_closed_when_any_request_wants_frame_interpolation() -> None:
    """RIFE interpolation needs float [B, C, F, H, W]; one such request closes the gate."""
    batch = [_sp("np"), _sp("np", interpolate=True)]
    assert _should_reduce_video_on_device(_od_config(reduce=True), batch, output_type="np") is False


# --- Fail-closed checks in the uint8 fast paths ----------------------------


def test_wan_post_process_refuses_uint8_when_interpolation_was_requested() -> None:
    """If the gate ever regresses, WAN must raise rather than drop interpolation."""
    frames = torch.randint(0, 256, (1, 2, 8, 8, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="frame interpolation"):
        get_wan22_post_process_func(SimpleNamespace())(
            frames, output_type="np", sampling_params=_sp("np", interpolate=True)
        )


def test_cosmos3_post_process_refuses_uint8_when_guardrails_are_enabled() -> None:
    """If the gate ever regresses, Cosmos3 must raise rather than skip the safety check."""
    post_process = get_cosmos3_post_process_func(SimpleNamespace(model_config={"guardrails": True}))
    video = torch.randint(0, 256, (1, 2, 8, 8, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="guardrails are enabled"):
        post_process({"video": video}, output_type="np", sampling_params=SimpleNamespace(extra_args={}))


def test_cosmos3_post_process_accepts_uint8_when_guardrails_are_disabled() -> None:
    """The counterpart: with guardrails off the uint8 video passes through."""
    post_process = get_cosmos3_post_process_func(SimpleNamespace(model_config={"guardrails": False}))
    video = torch.randint(0, 256, (1, 2, 8, 8, 3), dtype=torch.uint8)

    result = post_process({"video": video}, output_type="np", sampling_params=SimpleNamespace(extra_args={}))

    assert result.dtype.name == "uint8"
    assert result.shape == (1, 2, 8, 8, 3)
