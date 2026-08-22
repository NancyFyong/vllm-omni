# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the claim that this feature changes nothing until it is switched on.

Every device-side reduction and every alternative sink is opt-in, so a deployment
that does not configure ``video_output_transport`` must behave exactly as it did
before. That is the property a reviewer has to be able to check in one place.

The historical encoder options and the default response shape are pinned in
tests/entrypoints/test_serving_video_codec_options.py and
tests/entrypoints/test_video_output_transport_sinks.py respectively; this file
covers the config defaults and the gate.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig
from vllm_omni.diffusion.postprocess.device_reduction import should_enable_device_postprocess
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_transport_defaults() -> None:
    """Each default is spelled out, so changing one has to be deliberate."""
    transport = VideoOutputTransportConfig()

    assert transport.enable_device_postprocess is False
    assert transport.transport_mode == "base64"
    assert transport.output_format == "mp4"
    # None means "whatever the container needs"; a literal codec here would
    # override the format-derived choice and could pair h264 with webm.
    assert transport.video_codec is None
    # Empty means "use the codec-aware fast presets".
    assert transport.video_codec_options == {}
    assert transport.shm_threshold_bytes == 1_000_000


def test_a_default_diffusion_config_carries_a_default_transport() -> None:
    config = OmniDiffusionConfig(model="x")

    assert isinstance(config.video_output_transport, VideoOutputTransportConfig)
    assert config.video_output_transport.enable_device_postprocess is False


def _sp(output_type="np", interpolation=False):
    return OmniDiffusionSamplingParams(output_type=output_type, enable_frame_interpolation=interpolation)


def test_no_pipeline_reduces_under_a_default_config() -> None:
    """The six real call shapes, taken from the pipelines, must all decline.

    Keeping them together means a reviewer does not have to read six pipelines to
    confirm the feature is off by default.
    """
    config = OmniDiffusionConfig(model="x")
    sp = _sp()

    # wan2_2/pipeline_wan2_2.py: batch of requests plus the pipeline output type
    assert should_enable_device_postprocess(config, [sp, sp], output_type="np") is False
    # hunyuan_video/pipeline_hunyuan_video_1_5.py: config only, output is always PIL
    assert should_enable_device_postprocess(config) is False
    # ltx2/ltx2_runtime.py: pipeline output type only
    assert should_enable_device_postprocess(config, output_type="np") is False
    # minimax_h3/pipeline_minimax_h3.py: a single sampling object
    assert should_enable_device_postprocess(config, sp) is False
    # cosmos3/pipeline_cosmos3.py: guardrails veto, here inactive
    assert should_enable_device_postprocess(config, sp, blocked=False) is False
    # lingbot_video/pipeline_lingbot_video.py: text-to-image veto, here inactive
    assert should_enable_device_postprocess(config, output_type="np", blocked=False) is False


def test_the_flag_is_what_makes_the_difference() -> None:
    """The mirror of the test above: the same shapes reduce once it is enabled.

    Without this, "everything returns False" would also pass if the gate were
    broken and never reduced at all.
    """
    config = OmniDiffusionConfig(
        model="x",
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True),
    )
    sp = _sp()

    assert should_enable_device_postprocess(config, [sp, sp], output_type="np") is True
    assert should_enable_device_postprocess(config) is True
    assert should_enable_device_postprocess(config, output_type="np") is True
    assert should_enable_device_postprocess(config, sp) is True
    assert should_enable_device_postprocess(config, sp, blocked=False) is True
    assert should_enable_device_postprocess(config, output_type="np", blocked=False) is True


def test_every_video_pipeline_uses_the_shared_gate() -> None:
    """No pipeline may reintroduce a gate of its own.

    Six divergent copies of this decision is what the shared gate replaced, and
    a local copy would drift out of step with it unnoticed.
    """
    import inspect
    from pathlib import Path

    pipelines = {
        "wan2_2/pipeline_wan2_2.py",
        "hunyuan_video/pipeline_hunyuan_video_1_5.py",
        "ltx2/ltx2_runtime.py",
        "minimax_h3/pipeline_minimax_h3.py",
        "cosmos3/pipeline_cosmos3.py",
        "lingbot_video/pipeline_lingbot_video.py",
    }

    import vllm_omni.diffusion.models as models_pkg

    root = Path(inspect.getfile(models_pkg)).parent
    for relative in sorted(pipelines):
        source = (root / relative).read_text()
        # The trailing paren matters: the import alone would satisfy a plain
        # substring check even after the call site was replaced.
        assert "should_enable_device_postprocess(" in source, f"{relative} does not call the shared gate"
        assert "def _should_reduce" not in source, f"{relative} defines a local gate again"
