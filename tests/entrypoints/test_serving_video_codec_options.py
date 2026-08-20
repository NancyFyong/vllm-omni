# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video encoder resolution for the engine->HTTP and engine->WebSocket hops.

The codec and its options were constants duplicated across three encode sites,
with the streaming site silently using a different value. They are now resolved
in one place from ``video_output_transport`` with per-request overrides, and a
hardware encoder the host cannot open falls back to software.

The historical constants are pinned here: a regression that changes what the
default deployment sends to FFmpeg turns these red.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.diffusion.utils.media_utils import resolve_encoder_settings
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes, resolve_video_encoder_settings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# What the two encode sites hardcoded before this became config.
_HISTORICAL_HTTP = {"preset": "ultrafast", "threads": "0"}
_HISTORICAL_STREAM = {"preset": "ultrafast", "threads": "0", "tune": "zerolatency"}

# Exists in the FFmpeg build but cannot be opened here: Hopper data-center GPUs
# ship no NVENC engine, so this exercises the real fallback, not a simulated one.
_UNAVAILABLE_HW_CODEC = "h264_nvenc"


def _client(transport=None, *, callable_api: bool = True):
    """Engine client stub.

    ``callable_api=True`` mirrors production (``AsyncOmni`` exposes
    ``get_diffusion_od_config()``); ``False`` covers the plain ``od_config``
    attribute fallback.
    """
    od_config = SimpleNamespace(video_output_transport=transport) if transport is not None else None
    if callable_api:
        return SimpleNamespace(get_diffusion_od_config=lambda: od_config)
    return SimpleNamespace(od_config=od_config)


def _serving(transport=None, *, callable_api: bool = True) -> OmniOpenAIServingVideo:
    serving = object.__new__(OmniOpenAIServingVideo)
    serving._engine_client = _client(transport, callable_api=callable_api)
    return serving


# --- defaults preserved -----------------------------------------------------


@pytest.mark.parametrize("callable_api", [True, False], ids=["get_diffusion_od_config", "od_config_attr"])
def test_http_default_matches_historical_constant(callable_api: bool) -> None:
    serving = _serving(VideoOutputTransportConfig(), callable_api=callable_api)
    codec, options = serving._resolve_video_encoder(SimpleNamespace(extra_params=None))
    assert (codec, options) == ("h264", _HISTORICAL_HTTP)


def test_streaming_default_matches_historical_constant() -> None:
    """The streaming site used a different constant; low_latency must reproduce it."""
    client = _client(VideoOutputTransportConfig())
    assert tuple(resolve_video_encoder_settings(client, None, low_latency=True)) == ("h264", _HISTORICAL_STREAM)


def test_default_when_engine_exposes_no_config() -> None:
    serving = _serving()
    assert tuple(serving._resolve_video_encoder(SimpleNamespace(extra_params=None))) == ("h264", _HISTORICAL_HTTP)


# --- precedence -------------------------------------------------------------


def test_options_come_from_deployment_config() -> None:
    transport = VideoOutputTransportConfig(video_codec_options={"crf": "30"})
    serving = _serving(transport)
    assert tuple(serving._resolve_video_encoder(SimpleNamespace(extra_params=None))) == ("h264", {"crf": "30"})


def test_per_request_options_override_config() -> None:
    transport = VideoOutputTransportConfig(video_codec_options={"crf": "30"})
    serving = _serving(transport)
    request = SimpleNamespace(extra_params={"video_codec_options": {"preset": "medium"}})
    assert tuple(serving._resolve_video_encoder(request)) == ("h264", {"preset": "medium"})


def test_per_request_codec_override_is_honoured() -> None:
    serving = _serving(VideoOutputTransportConfig())
    request = SimpleNamespace(extra_params={"video_codec": "libx264"})
    codec, options = serving._resolve_video_encoder(request)
    assert codec == "libx264"
    assert options == _HISTORICAL_HTTP


# --- fallback for an encoder this host cannot open ---------------------------


def test_unavailable_hardware_codec_falls_back_to_software() -> None:
    transport = VideoOutputTransportConfig(video_codec=_UNAVAILABLE_HW_CODEC)
    serving = _serving(transport)
    assert tuple(serving._resolve_video_encoder(SimpleNamespace(extra_params=None))) == ("h264", _HISTORICAL_HTTP)


def test_fallback_also_drops_the_requested_codecs_options() -> None:
    """Keeping them would fail the encode: FFmpeg rejects another family's options."""
    codec, options = resolve_encoder_settings(_UNAVAILABLE_HW_CODEC, {"preset": "p1", "tune": "ull"})
    assert codec == "h264"
    assert options == _HISTORICAL_HTTP


def test_unknown_codec_falls_back_instead_of_raising() -> None:
    assert resolve_encoder_settings("no_such_encoder", None) == ("h264", _HISTORICAL_HTTP)


def test_fallback_still_produces_a_decodable_mp4() -> None:
    """End-to-end proof: requesting an unopenable encoder still returns video."""
    av = pytest.importorskip("av")
    frames = np.zeros((6, 32, 48, 3), dtype=np.uint8)

    codec, options = resolve_encoder_settings(_UNAVAILABLE_HW_CODEC, None)
    mp4 = _encode_video_bytes(frames, fps=8, video_codec=codec, video_codec_options=options)

    with av.open(__import__("io").BytesIO(mp4)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.width == 48
        assert stream.codec_context.height == 32


def test_configured_codec_that_is_available_is_used_verbatim() -> None:
    """The fallback must not fire for a usable encoder."""
    assert resolve_encoder_settings("libx264", {"crf": "20"}) == ("libx264", {"crf": "20"})
