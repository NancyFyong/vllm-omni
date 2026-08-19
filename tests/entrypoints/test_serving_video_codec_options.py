# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MP4 codec-option resolution for the engine->HTTP hop in serving_video.

The MP4 encoder options were a constant duplicated across the two encode sites.
They are now first-class config (``video_output_transport.video_codec_options``)
resolved with a clear precedence: per-request override > deployment config >
default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HISTORICAL_DEFAULT = {"preset": "ultrafast", "threads": "0"}


def _make_serving(video_codec_options=None) -> OmniOpenAIServingVideo:
    transport = (
        VideoOutputTransportConfig(video_codec_options=video_codec_options)
        if video_codec_options is not None
        else VideoOutputTransportConfig()
    )
    od_config = SimpleNamespace(video_output_transport=transport)
    engine_client = SimpleNamespace(od_config=od_config)
    serving = object.__new__(OmniOpenAIServingVideo)
    serving._engine_client = engine_client
    return serving


def test_codec_options_default_matches_historical_constant():
    serving = _make_serving()
    request = SimpleNamespace(extra_params=None)
    assert serving._resolve_video_codec_options(request) == _HISTORICAL_DEFAULT


def test_codec_options_come_from_deployment_config():
    serving = _make_serving({"preset": "p7", "threads": "4"})
    request = SimpleNamespace(extra_params=None)
    assert serving._resolve_video_codec_options(request) == {"preset": "p7", "threads": "4"}


def test_per_request_codec_options_override_config():
    serving = _make_serving({"preset": "p7", "threads": "4"})
    request = SimpleNamespace(extra_params={"video_codec_options": {"preset": "medium"}})
    # Per-request override wins over the deployment config.
    assert serving._resolve_video_codec_options(request) == {"preset": "medium"}


def test_codec_options_default_when_engine_has_no_config():
    serving = object.__new__(OmniOpenAIServingVideo)
    serving._engine_client = SimpleNamespace()  # no od_config attribute
    request = SimpleNamespace(extra_params=None)
    assert serving._resolve_video_codec_options(request) == _HISTORICAL_DEFAULT
