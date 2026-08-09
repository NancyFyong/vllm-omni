# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the JoyAI-Video-Edit pipeline's request-shaping helpers.

Everything here runs on the CPU with no weights. These functions sit between the caller and a 30 GiB
model, so their job is to turn an unusable request into an error *before* anything is scheduled --
which means the error messages are part of the contract, not decoration, and are asserted as such.

Several of these tests are regression guards for bugs this port actually shipped and then fixed.
Three were caught locally: ``resolve_num_frames`` reported a lower bracket above the value it was
rejecting; ``_resize_video_uint8`` inherited a nearest-neighbour filter where upstream resizes
bicubic; and the post-process envelope was read through a field ``OmniRequestOutput`` does not have.

Two more were caught only by measuring per-frame agreement against upstream's own run on the same
box, weights and seed, and both are worth knowing about because neither is visible from inside the
process: the source encode shared its generator with the denoise rollout (``resolve_generators``), and
the resize used torch's bicubic kernel rather than PIL's (``_resize_video_uint8``). Each produced a
perfectly plausible edited video while disagreeing with upstream on every frame. Together they cost
7-11 dB of cold-start fidelity, so the tests guarding them assert against upstream's exact expression
rather than against self-consistency.
"""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION,
)
from vllm_omni.diffusion.models.joyai_video_edit.pipeline_joyai_video_edit import (
    DEFAULT_JOYAI_VIDEO_EDIT_FPS,
    JoyAIVideoEditPipeline,
    _normalize_frame_array,
    _resize_video_uint8,
    get_joyai_video_edit_post_process_func,
    get_joyai_video_edit_pre_process_func,
    normalize_dit_state_dict,
    resolve_generators,
    resolve_num_frames,
    resolve_resolution,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _clip(num_frames: int = 9, height: int = 48, width: int = 72) -> np.ndarray:
    """A tiny gradient clip; contents only matter where a test says so."""
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    frame = np.broadcast_to(ramp, (height, width))
    return np.repeat(np.repeat(frame[None, ..., None], num_frames, axis=0), 3, axis=-1).astype(np.uint8)


# --- frame-count resolution ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("available", "expected"),
    [(9, 9), (16, 9), (17, 17), (24, 17), (73, 73), (100, 97), (1, 1), (8, 1)],
)
def test_unconstrained_request_trims_to_the_longest_valid_prefix(available, expected):
    """The source's length is an input, not an instruction, so it is trimmed rather than rejected."""
    assert resolve_num_frames(available, None) == expected


@pytest.mark.parametrize("requested", [0, None, 1])
def test_falsy_frame_counts_mean_unconstrained(requested):
    """1 is included on purpose. ``OmniDiffusionSamplingParams.num_frames`` defaults to 1 for image
    models, so honouring it literally would make every request that does not mention a frame count
    return a single frame -- which is what this integration did until the default was traced."""
    assert resolve_num_frames(100, requested) == 97


def test_default_frame_count_is_reinterpreted_out_loud(caplog):
    """The reinterpretation is defensible only because it is logged; an unexplained 97-frame result
    from ``num_frames=1`` is worse than a 1-frame one."""
    with caplog.at_level("INFO", logger="vllm_omni.diffusion.models.joyai_video_edit.pipeline_joyai_video_edit"):
        assert resolve_num_frames(100, 1) == 97
    assert "97 of 100" in caplog.text


def test_a_single_frame_source_still_resolves_to_one():
    """No log line and no error when 1 is all the source can give."""
    assert resolve_num_frames(1, 1) == 1


def test_requesting_more_frames_than_the_source_has_is_rejected():
    """This model edits a clip and cannot extend one. Trimming the *request* down would quietly
    return a shorter video than was asked for."""
    with pytest.raises(ValueError, match="cannot extend"):
        resolve_num_frames(9, 17)


@pytest.mark.parametrize(
    ("requested", "expected_bracket"),
    [(72, "65 and 73"), (74, "73 and 81"), (100, "97 and 105"), (2, "1 and 9"), (8, "1 and 9")],
)
def test_invalid_frame_count_names_a_bracket_that_contains_it(requested, expected_bracket):
    """Regression: the lower bound was computed as ``1 + 8 * (n // 8)``, which for 72 suggests
    "73 and 81" -- a lower bracket *above* the rejected value, sending the caller upward when the
    only value that fits their clip is downward."""
    with pytest.raises(ValueError, match=expected_bracket) as excinfo:
        resolve_num_frames(1000, requested)
    lower, upper = (int(part) for part in expected_bracket.split(" and "))
    assert lower <= requested <= upper, f"{expected_bracket} does not bracket {requested}"
    assert f"1 + {VAE_TEMPORAL_COMPRESSION}n" in str(excinfo.value)


def test_empty_source_is_rejected():
    with pytest.raises(ValueError, match="no frames"):
        resolve_num_frames(0, None)


# --- resolution ---------------------------------------------------------------------------------


def test_resolution_defaults_to_the_shipped_reference():
    assert resolve_resolution(1080, 1920, None, None) == (DEFAULT_HEIGHT, DEFAULT_WIDTH)


def test_defaults_are_themselves_valid():
    """A guard on the constants, not on this function: a default that violated the rule would make
    every unparameterised request fail."""
    assert DEFAULT_HEIGHT % VAE_SPATIAL_COMPRESSION == 0
    assert DEFAULT_WIDTH % VAE_SPATIAL_COMPRESSION == 0


def test_width_1280_is_rejected_and_the_message_says_why():
    """1280 divides 16 but not 24, so it bypasses the VAE ``Stem`` and produces a wrong-shaped latent
    with no error anywhere downstream. It is the size everyone reaches for, so the message has to
    name it and point at 1248."""
    with pytest.raises(ValueError, match="1280 is the common trap") as excinfo:
        resolve_resolution(720, 1280, 720, 1280)
    assert "width=1280" in str(excinfo.value)
    assert str(DEFAULT_WIDTH) in str(excinfo.value)


def test_both_bad_dimensions_are_reported_together():
    """Reporting only the first would make the caller fix one, re-run the 30 GiB load, and fail
    again on the other."""
    with pytest.raises(ValueError) as excinfo:
        resolve_resolution(100, 100, 700, 1250)
    assert "height=700" in str(excinfo.value) and "width=1250" in str(excinfo.value)


def test_valid_non_default_resolution_is_accepted():
    assert resolve_resolution(0, 0, 480, 864) == (480, 864)


# --- resize -------------------------------------------------------------------------------------


def test_downscale_is_bicubic_not_nearest():
    """Regression: this was ``mode="nearest"``, factored in from a sibling pipeline, while upstream
    resizes with PIL BICUBIC. Nearest-neighbour subsampling of a ramp reproduces a subset of the
    source values exactly; a filtered resize averages neighbours, so the output contains values that
    appear nowhere in the input row.
    """
    width = 96
    ramp = np.arange(width, dtype=np.uint8) * 2
    frames = np.repeat(np.broadcast_to(ramp, (1, 24, width))[..., None], 3, axis=-1).astype(np.uint8)

    out = _resize_video_uint8(frames, height=24, width=width // 4)
    assert out.shape == (1, 24, width // 4, 3)
    row = out[0, 12, :, 0].astype(int)
    assert (np.diff(row) > 0).all(), "a monotone ramp must stay monotone"
    # Nearest would emit only multiples of 8 (every 4th value of a step-2 ramp).
    assert any(value % 8 != 0 for value in row[1:-1]), f"all interior values are nearest-grid: {row}"


def test_resize_short_circuits_when_the_size_already_matches():
    """Not an optimisation: a bicubic round-trip at the same size is not the identity, so re-filtering
    a clip that needs no resize would soften every frame of every request."""
    frames = _clip(num_frames=3, height=48, width=72)
    assert _resize_video_uint8(frames, height=48, width=72) is frames


def test_resize_matches_upstreams_pil_bicubic_bit_for_bit():
    """Fidelity regression, and the reason it needs asserting at all: this shipped as
    ``F.interpolate(mode="bicubic", antialias=True)``, chosen as "the closest torch equivalent" to the
    ``PIL.Image.BICUBIC`` upstream uses in ``joyomni_streaming.py::_resize_frame``. It is not close
    enough. PIL's bicubic is Catmull-Rom (``a = -0.5``); torch's is ``a = -0.75``. Feeding the DiT
    conditioning pixels that differ from upstream's shifts the entire clip by a constant amount which
    reads as a plausible edit rather than a bug, and which is only visible as a *flat* few-dB
    disagreement with upstream from frame 0 onward -- i.e. it survives every shape, range and
    monotonicity check in this file.

    Asserted against the expression written out independently rather than against a stored array, so
    it also pins the parts a PIL rewrite can still get wrong: the ``(width, height)`` argument order
    (checked here on a non-square target, which is what makes a swap fail rather than silently pass),
    per-frame independence, and uint8 in/out.
    """
    from PIL import Image

    # Deliberately NOT the ramp `_clip` builds: every cubic convolution kernel reproduces a linear
    # function exactly regardless of its `a` constant, so a ramp is bit-identical under both filters
    # and would make the discriminating assertion below pass vacuously. Noise has the high-frequency
    # content the two kernels weight differently.
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(3, 12, 20, 3), dtype=np.uint8)
    # 12x20 -> 24x40 is an *upscale*, where torch ignores `antialias` outright, so the kernel constant
    # is the only thing left to differ. The showcase geometry (832x480 -> 1248x720) is this case.
    out = _resize_video_uint8(frames, height=24, width=40)
    assert out.shape == (3, 24, 40, 3)
    assert out.dtype == np.uint8

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    for index, frame in enumerate(frames):
        expected = np.asarray(Image.fromarray(frame).resize((40, 24), resampling), dtype=np.uint8)
        np.testing.assert_array_equal(out[index], expected)

    torch_bicubic = (
        torch.nn.functional.interpolate(
            torch.from_numpy(frames).permute(0, 3, 1, 2).float(),
            size=(24, 40),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .permute(0, 2, 3, 1)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    assert not np.array_equal(out, torch_bicubic), (
        "PIL and torch bicubic agreed here, so this test can no longer tell the two filters apart "
        "and would not have caught the regression it exists for"
    )


# --- RNG stream separation ------------------------------------------------------------------------


def test_source_encode_does_not_consume_the_chunk_noise_stream():
    """Fidelity regression, and the one that cost the most to find. Upstream keeps two *independent*
    random streams: the VAE posterior sample is drawn from the global RNG (``latent_dist.sample()``
    with no generator, ``models/pipeline.py:315``) while per-chunk denoise noise comes from a
    separately seeded ``torch.Generator`` (``joyomni_streaming.py:385``). This port originally passed
    the request's single generator to both, so encoding the source advanced the Philox stream by one
    large draw *per latent frame* -- 15 on a 113-frame clip -- before the rollout took its first
    sample. Every chunk was then denoised from noise upstream never used, which is a whole-clip
    divergence that still produces a plausible edit: the output looks like a slightly different take,
    not like a bug, and no shape, dtype, finiteness or range check fires. It measured as a flat ~18 dB
    agreement with upstream from frame 0 onward, and fixing it moved cold-start agreement on the
    showcase cases by 3.5-8.5 dB.

    The encode side draws a deliberately *variable* number of times, because the failure mode is
    precisely that the draw count leaks into the other stream. Comparing the rollout's first sample
    against a fresh ``manual_seed(seed)`` is what makes this a fidelity claim rather than a
    self-consistency one: it asserts the rollout's first draw is the first draw off the seed, which is
    what upstream's is.
    """
    seed = 42
    noise_generator, encode_generator = resolve_generators(OmniDiffusionSamplingParams(seed=seed), torch.device("cpu"))
    assert encode_generator is not noise_generator

    for _ in range(15):  # stands in for one posterior sample per latent frame
        torch.randn(8, generator=encode_generator)

    first_chunk_noise = torch.randn(8, generator=noise_generator)
    expected = torch.randn(8, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(first_chunk_noise, expected), (
        "the rollout's first noise draw is not the first draw off the seed, so the source encode has "
        "advanced the stream the chunk noise comes from"
    )


def test_both_streams_start_from_the_same_seed():
    """The two streams are independent but not *different*: upstream's encode and rollout both begin
    at a seed-42 state (its encode from ``seed_everything(42)``, its rollout from an explicit
    generator). Deriving the encode stream from, say, ``seed + 1`` would fix the leak and still feed
    the DiT a different source latent than upstream computes.
    """
    noise_generator, encode_generator = resolve_generators(OmniDiffusionSamplingParams(seed=7), torch.device("cpu"))
    assert torch.equal(torch.randn(8, generator=noise_generator), torch.randn(8, generator=encode_generator))


def test_an_unseeded_request_leaves_both_streams_on_the_global_rng():
    """No seed means the caller declined reproducibility, and inventing one here would silently make
    every unseeded request produce the same video."""
    assert resolve_generators(OmniDiffusionSamplingParams(), torch.device("cpu")) == (None, None)


def test_a_caller_supplied_generator_is_used_for_noise_rather_than_reseeded():
    """``sampling_params.generator`` may arrive already advanced -- that is the point of passing an
    object instead of a seed -- so the rollout must continue *that* stream. Re-deriving it from
    ``initial_seed()`` (which is what the encode side does) would discard the caller's position.
    """
    supplied = torch.Generator().manual_seed(5)
    torch.randn(8, generator=supplied)
    expected = torch.randn(8, generator=supplied.clone_state())

    noise_generator, _ = resolve_generators(OmniDiffusionSamplingParams(generator=supplied), torch.device("cpu"))
    assert noise_generator is supplied
    assert torch.equal(torch.randn(8, generator=noise_generator), expected)


# --- frame layout normalisation ------------------------------------------------------------------


def test_normalized_frames_always_carry_a_time_axis():
    """The divergence from the sibling pipeline this was factored from: that one returns a bare array
    whose callers then index ``[0]``, which on a ``[T, H, W, 3]`` clip silently selects frame 0."""
    assert _normalize_frame_array(np.zeros((48, 72, 3), np.uint8)).shape == (1, 48, 72, 3)
    assert _normalize_frame_array(np.zeros((5, 48, 72, 3), np.uint8)).shape == (5, 48, 72, 3)


def test_channel_first_layouts_are_transposed():
    assert _normalize_frame_array(np.zeros((3, 48, 72), np.uint8)).shape == (1, 48, 72, 3)
    assert _normalize_frame_array(np.zeros((3, 5, 48, 72), np.uint8)).shape == (5, 48, 72, 3)


def test_unusable_rank_is_rejected():
    with pytest.raises(ValueError, match=r"\[T, H, W, C\]"):
        _normalize_frame_array(np.zeros((5, 6), np.uint8))


# --- checkpoint unwrapping -----------------------------------------------------------------------


def test_bare_state_dict_is_passed_through():
    sd = {"img_in.weight": torch.zeros(1), "norm_out.bias": torch.zeros(1)}
    assert normalize_dit_state_dict(sd) == pytest.approx(sd)


@pytest.mark.parametrize("wrapper", ["module", "model", "state_dict", "ema_state_dict"])
def test_training_harness_wrappers_are_unwrapped(wrapper):
    inner = {"img_in.weight": torch.zeros(1)}
    assert list(normalize_dit_state_dict({wrapper: inner, "epoch": 7})) == ["img_in.weight"]


@pytest.mark.parametrize("prefix", ["model.module.", "module.model.", "transformer.", "module.", "model."])
def test_uniform_prefixes_are_stripped_whole(prefix):
    """``model.module.`` must be tried before ``module.`` or ``model.``, or one pass leaves a dangling
    half-prefix behind and ``strict=True`` reports every key as unexpected."""
    sd = {f"{prefix}img_in.weight": torch.zeros(1), f"{prefix}blocks.0.img_mod.weight": torch.zeros(1)}
    assert sorted(normalize_dit_state_dict(sd)) == ["blocks.0.img_mod.weight", "img_in.weight"]


def test_a_prefix_shared_by_only_some_keys_is_left_alone():
    """``model.`` is also a legitimate leading component of a real parameter path. Stripping it from
    the subset that happens to start with it would rename half the tree; leaving it produces a
    ``strict=True`` error naming the keys, which is the diagnosable failure."""
    sd = {"model.img_in.weight": torch.zeros(1), "blocks.0.weight": torch.zeros(1)}
    assert sorted(normalize_dit_state_dict(sd)) == ["blocks.0.weight", "model.img_in.weight"]


# --- pre-process ---------------------------------------------------------------------------------


def _request(video, **params) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "make it snow", "multi_modal_data": {"video": video}},
        sampling_params=OmniDiffusionSamplingParams(**params),
        request_id="joyai-unit-0",
    )


def test_pre_process_pins_geometry_onto_the_sampling_params():
    """The worker reads geometry from ``sampling_params``, not from the frames, so admission has to
    write back what it resolved -- otherwise the pipeline re-derives it against a clip that has
    already been trimmed and resized."""
    request = _request(_clip(num_frames=20, height=48, width=72), height=48, width=72)
    out = get_joyai_video_edit_pre_process_func(None)(request)

    assert out.sampling_params.num_frames == 17
    assert (out.sampling_params.height, out.sampling_params.width) == (48, 72)
    assert out.prompt["multi_modal_data"]["video"].shape == (17, 48, 72, 3)


def test_pre_process_resizes_to_the_requested_size():
    request = _request(_clip(num_frames=9, height=96, width=144), height=48, width=72)
    out = get_joyai_video_edit_pre_process_func(None)(request)
    assert out.prompt["multi_modal_data"]["video"].shape == (9, 48, 72, 3)


def test_pre_process_records_a_missing_source_fps_as_none():
    """An ndarray input carries no frame rate. Recording ``None`` rather than omitting the key keeps
    the pipeline's fallback chain (request -> source -> default) reading one place."""
    request = _request(_clip(), height=48, width=72)
    out = get_joyai_video_edit_pre_process_func(None)(request)
    assert out.prompt["additional_information"]["source_video_fps"] is None


def test_pre_process_rejects_bad_geometry_at_admission():
    """The whole point of the pre-process hook: fail here, not after the DiT is resident."""
    with pytest.raises(ValueError, match="1280 is the common trap"):
        get_joyai_video_edit_pre_process_func(None)(_request(_clip(), height=720, width=1280))


# --- post-process --------------------------------------------------------------------------------


def test_post_process_envelope_carries_the_video_and_its_fps():
    """The engine lifts ``metadata["video"]["fps"]`` to a top-level ``fps`` on ``multimodal_output``
    and routes ``payload["video"]`` to ``OmniRequestOutput.images``. Naming either key differently
    loses it with no error."""
    video = _clip(num_frames=9)
    result = get_joyai_video_edit_post_process_func(None)((video, 24.0))
    assert result["payload"]["video"] is video
    assert result["metadata"]["video"]["fps"] == 24.0


def test_post_process_supplies_a_default_fps_for_a_bare_video():
    result = get_joyai_video_edit_post_process_func(None)(_clip())
    assert result["metadata"]["video"]["fps"] == DEFAULT_JOYAI_VIDEO_EDIT_FPS


def test_latent_output_type_bypasses_the_envelope():
    """``output_type="latent"`` is a debugging escape hatch; wrapping the tensor would make callers
    unwrap a payload to reach a latent that has no fps."""
    latents = torch.zeros(1, 64, 2, 3, 4)
    assert get_joyai_video_edit_post_process_func(None)((latents, 24.0), output_type="latent") is latents


@pytest.mark.parametrize(("output_type", "check"), [("pil", "pil"), ("pt", "pt"), ("tensor", "pt")])
def test_alternate_output_types(output_type, check):
    result = get_joyai_video_edit_post_process_func(None)((_clip(num_frames=4), 24.0), output_type=output_type)
    video = result["payload"]["video"]
    if check == "pil":
        assert len(video) == 4 and video[0].size == (72, 48)
    else:
        assert isinstance(video, torch.Tensor) and video.shape == (4, 3, 48, 72)


def test_sampling_params_output_type_wins_over_the_argument():
    """The engine inspects the signature and passes ``sampling_params`` but never ``output_type``, so
    a request-level ``output_type`` only takes effect through this override."""
    result = get_joyai_video_edit_post_process_func(None)(
        (_clip(num_frames=2), 24.0), sampling_params=OmniDiffusionSamplingParams(output_type="pt")
    )
    assert isinstance(result["payload"]["video"], torch.Tensor)


# --- batch guard ---------------------------------------------------------------------------------


def test_forward_refuses_more_than_one_prompt():
    """The DiT shares one rotary table and one KV scope across the batch, so samples 1..N would be
    generated against sample 0's positions and returned as if they were correct.

    Nothing routine reaches this: ``_max_num_seqs`` defaults to 1 engine-wide, so the engine serialises
    requests and ``tests/e2e/offline_inference/test_joyai_video_edit.py`` asserts that instead. This
    guard is for the two ways past that -- a caller who raises ``max_num_seqs``, and one who builds a
    ``DiffusionRequestBatch`` directly -- which is exactly why it needs a test of its own rather than
    being assumed covered by the cap.

    Asserted on an uninitialised instance on purpose: it has to fire before any component is touched,
    which is also what makes it testable without 47 GiB of weights.
    """
    pipe = JoyAIVideoEditPipeline.__new__(JoyAIVideoEditPipeline)
    torch.nn.Module.__init__(pipe)
    requests = [_request(_clip(), height=48, width=72), _request(_clip(), height=48, width=72)]
    with pytest.raises(ValueError, match="one video per request, got 2"):
        pipe.forward(DiffusionRequestBatch(requests=requests))
