# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the six per-model gates the shared gate replaced.

Before unification each video pipeline decided on its own whether a decoded video
could be reduced to uint8 before the D2H copy. Those six expressions are
reproduced verbatim below and cross-checked against
``should_enable_device_postprocess`` over every combination of their inputs, so
the refactor cannot quietly change which requests take the reduced path.

Where the shared gate deliberately differs, the divergence is listed in
``_EXPECTED_DIVERGENCES`` rather than papered over.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.diffusion.postprocess.device_reduction import should_enable_device_postprocess
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# --- the six historical gates, copied from the pre-unification pipelines -----
#
# Only ``reduce_video_on_device`` was renamed to ``enable_device_postprocess``;
# the logic is untouched.


def _historical_wan(od_config, sampling_params_list, *, output_type):
    transport = getattr(od_config, "video_output_transport", None)
    return (
        output_type == "np"
        and transport is not None
        and transport.enable_device_postprocess
        and all((sp.output_type or "np") == "np" for sp in sampling_params_list)
        and not any(getattr(sp, "enable_frame_interpolation", False) for sp in sampling_params_list)
    )


def _historical_hunyuan(od_config):
    return od_config.video_output_transport.enable_device_postprocess


def _historical_ltx2(od_config, *, output_type):
    transport = getattr(od_config, "video_output_transport", None)
    return transport is not None and transport.enable_device_postprocess and output_type == "np"


def _historical_minimax(od_config, sampling):
    transport = getattr(od_config, "video_output_transport", None)
    return (
        transport is not None
        and transport.enable_device_postprocess
        and getattr(sampling, "output_type", None) in (None, "np")
    )


def _historical_cosmos3(od_config, sp, *, guardrails_enabled):
    transport = getattr(od_config, "video_output_transport", None)
    return (
        transport is not None
        and transport.enable_device_postprocess
        and getattr(sp, "output_type", None) in (None, "np")
        and not guardrails_enabled
    )


def _historical_lingbot(od_config, *, output_type, is_t2i):
    transport = getattr(od_config, "video_output_transport", None)
    return output_type == "np" and not is_t2i and transport is not None and transport.enable_device_postprocess


# --- input space --------------------------------------------------------------

_FLAGS = (False, True)
_OUTPUT_TYPES = ("np", "pil", "latent", None)


def _od(flag: bool | None):
    """A config whose transport is missing, off, or on."""
    if flag is None:
        return SimpleNamespace(video_output_transport=None)
    return SimpleNamespace(video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=flag))


def _sp(output_type, interpolation=False):
    return OmniDiffusionSamplingParams(output_type=output_type, enable_frame_interpolation=interpolation)


# MiniMax-H3 and Cosmos3 never look at enable_frame_interpolation: only the
# wan2_2 pipelines use interpolate_video_tensor. The shared gate checks it for
# every model, so for those two it can only decline a reduction that would have
# been safe. That forgoes an optimisation and never changes pixels.
_EXPECTED_DIVERGENCES = {"minimax_h3-interpolation", "cosmos3-interpolation"}


@pytest.mark.parametrize("flag", (None, *_FLAGS))
@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
@pytest.mark.parametrize("interpolation", _FLAGS)
def test_wan_gate_matches_history(flag, output_type, interpolation) -> None:
    od = _od(flag)
    batch = [_sp(output_type, interpolation), _sp("np", False)]
    assert should_enable_device_postprocess(od, batch, output_type=output_type) is _historical_wan(
        od, batch, output_type=output_type
    )


@pytest.mark.parametrize("flag", (None, *_FLAGS))
def test_hunyuan_gate_matches_history(flag) -> None:
    od = _od(flag)
    # HunyuanVideo dereferenced the transport directly, so a missing transport was
    # an AttributeError rather than False; the shared gate is the safer of the two.
    expected = False if flag is None else _historical_hunyuan(od)
    assert should_enable_device_postprocess(od) is expected


@pytest.mark.parametrize("flag", (None, *_FLAGS))
@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
def test_ltx2_gate_matches_history(flag, output_type) -> None:
    od = _od(flag)
    assert should_enable_device_postprocess(od, output_type=output_type) is _historical_ltx2(
        od, output_type=output_type
    )


@pytest.mark.parametrize("flag", (None, *_FLAGS))
@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
@pytest.mark.parametrize("interpolation", _FLAGS)
def test_minimax_gate_matches_history(flag, output_type, interpolation) -> None:
    od = _od(flag)
    sampling = _sp(output_type, interpolation)
    unified = should_enable_device_postprocess(od, sampling)
    historical = _historical_minimax(od, sampling)
    if unified != historical:
        assert interpolation, "the only accepted divergence is the interpolation check"
        assert "minimax_h3-interpolation" in _EXPECTED_DIVERGENCES
        assert historical is True and unified is False, "the gate may only become stricter"


@pytest.mark.parametrize("flag", (None, *_FLAGS))
@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
@pytest.mark.parametrize("interpolation", _FLAGS)
@pytest.mark.parametrize("guardrails", _FLAGS)
def test_cosmos3_gate_matches_history(flag, output_type, interpolation, guardrails) -> None:
    od = _od(flag)
    sp = _sp(output_type, interpolation)
    unified = should_enable_device_postprocess(od, sp, blocked=guardrails)
    historical = _historical_cosmos3(od, sp, guardrails_enabled=guardrails)
    if unified != historical:
        assert interpolation, "the only accepted divergence is the interpolation check"
        assert "cosmos3-interpolation" in _EXPECTED_DIVERGENCES
        assert historical is True and unified is False, "the gate may only become stricter"


@pytest.mark.parametrize("flag", (None, *_FLAGS))
@pytest.mark.parametrize("output_type", _OUTPUT_TYPES)
@pytest.mark.parametrize("is_t2i", _FLAGS)
def test_lingbot_gate_matches_history(flag, output_type, is_t2i) -> None:
    od = _od(flag)
    assert should_enable_device_postprocess(od, output_type=output_type, blocked=is_t2i) is _historical_lingbot(
        od, output_type=output_type, is_t2i=is_t2i
    )


def test_divergences_are_only_ever_stricter() -> None:
    """No input may make the shared gate reduce where a model previously did not.

    A stricter gate forgoes an optimisation; a looser one would reduce a video
    some pipeline still needs as float.
    """
    for flag, output_type, interp, guardrails, is_t2i in itertools.product(
        (None, *_FLAGS), _OUTPUT_TYPES, _FLAGS, _FLAGS, _FLAGS
    ):
        od = _od(flag)
        sp = _sp(output_type, interp)
        batch = [sp]

        pairs = [
            (
                should_enable_device_postprocess(od, batch, output_type=output_type),
                _historical_wan(od, batch, output_type=output_type),
            ),
            (
                should_enable_device_postprocess(od, output_type=output_type),
                _historical_ltx2(od, output_type=output_type),
            ),
            (should_enable_device_postprocess(od, sp), _historical_minimax(od, sp)),
            (
                should_enable_device_postprocess(od, sp, blocked=guardrails),
                _historical_cosmos3(od, sp, guardrails_enabled=guardrails),
            ),
            (
                should_enable_device_postprocess(od, output_type=output_type, blocked=is_t2i),
                _historical_lingbot(od, output_type=output_type, is_t2i=is_t2i),
            ),
        ]
        for unified, historical in pairs:
            if unified and not historical:
                raise AssertionError(
                    "shared gate reduces where the model did not: "
                    f"flag={flag} output_type={output_type!r} interpolation={interp} "
                    f"guardrails={guardrails} is_t2i={is_t2i}"
                )
