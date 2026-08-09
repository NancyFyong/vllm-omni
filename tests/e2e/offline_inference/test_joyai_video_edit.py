# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for JoyAI-Video-Edit offline video editing.

Nine frames, which is ``1 + 8 * 1`` -> two latent chunks. That is the smallest clip for which the
bounded KV window does anything at all: chunk 0 writes history and chunk 1 reads it, so a rollout that
never stores its clean-latent KV -- the third of the three DiT forwards per chunk, and the easiest one
to lose in a port -- produces different output here. A single-chunk clip would not notice.

What this test does *not* cover, deliberately:

- Cross-chunk drift. It needs ten or so chunks before accumulation separates from noise, which is a
  minutes-long run at this resolution. Kept as a manual check (``examples/offline_inference/
  video_to_video/README.md``), not paid for on every merge.
- Whether the sliding-window source encode matters. At nine frames the windowed and whole-clip encodes
  coincide *by construction* -- the first window is the whole clip -- so an assertion here would pass
  against an implementation that dropped the window entirely. Seventeen frames is the first length at
  which they differ; ``tests/diffusion/models/joyai_video_edit/test_vae.py`` covers the mechanism on a
  tiny config instead.
- Edit quality. There is no automated oracle for "did it follow the instruction", so this asserts the
  envelope and the geometry and leaves semantics to human review.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion]

MODEL_PATH_ENV = "JOYAI_VIDEO_EDIT_MODEL_DIR"
MODEL_CLASS = "JoyAIVideoEditPipeline"
PROMPT = "make the scene snowy"

HEIGHT, WIDTH = 720, 1248  # both divisible by 24; 1280 would bypass the VAE Stem
NUM_FRAMES = 9  # 1 + 8*1 -> 2 latent chunks
NUM_INFERENCE_STEPS = 2  # the whole AR-DMD-distilled schedule
SEED = 42


def _require_model_path() -> str:
    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path or not Path(model_path).is_dir():
        pytest.skip(
            f"JoyAI-Video-Edit weights not found. Set {MODEL_PATH_ENV} to a local directory prepared "
            f"by examples/offline_inference/video_to_video/download_joyai_video_edit.py "
            f"(a Hugging Face repo id will not work: the DiT is a raw .pth, not a diffusers subfolder)."
        )
    return model_path


def _source_clip() -> np.ndarray:
    """A drifting texture with a translating bright block.

    Synthetic rather than a checked-in asset, but not flat: the texture gives the VAE something to
    round-trip and the block gives motion that survives 8x temporal compression, so "the output
    ignores the source" is visible in the frames this test writes to the pytest tmp dir.
    """
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    frames = np.empty((NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    for t in range(NUM_FRAMES):
        base = 120 + 55 * np.sin(xx / 90.0 + t * 0.09) + 45 * np.cos(yy / 70.0 - t * 0.06)
        for channel, tint in enumerate((0.0, 12.0, 24.0)):
            frames[t, :, :, channel] = np.clip(base + tint, 0, 255).astype(np.uint8)
        left = 60 + t * 15
        frames[t, 260:460, left : left + 180] = 235
    return frames


def _sampling_params(**overrides) -> OmniDiffusionSamplingParams:
    params = {
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "seed": SEED,
        "output_type": "np",
    }
    params.update(overrides)
    return OmniDiffusionSamplingParams(**params)


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_video_to_video_001():
    """One request end to end: correct payload envelope, geometry, dtype, and a live source signal."""
    model_path = _require_model_path()
    source = _source_clip()

    with OmniRunner(model_path, model_class_name=MODEL_CLASS) as runner:
        outputs = runner.omni.generate(
            {"prompt": PROMPT, "multi_modal_data": {"video": source}},
            _sampling_params(),
        )

    assert outputs, "JoyAI-Video-Edit returned no outputs"
    result = outputs[0]

    # `payload["video"]` is routed to `.images`; there is no `.metadata` on OmniRequestOutput.
    assert getattr(result, "images", None), "no video frames returned"
    video = result.images[0]
    assert isinstance(video, np.ndarray), f"expected an ndarray, got {type(video)}"
    assert video.shape == (NUM_FRAMES, HEIGHT, WIDTH, 3), video.shape
    assert video.dtype == np.uint8, video.dtype
    assert np.isfinite(video.astype(np.float32)).all()

    # `metadata["video"]["fps"]` is lifted to a top-level `fps` by the output formatter. Asserting it
    # here is what catches a rename of either key, which would otherwise drop the value silently.
    multimodal_output = getattr(result, "multimodal_output", None) or {}
    assert float(multimodal_output["fps"]) > 0

    # Not a quality claim. A constant frame, or one that ignores the source's own dynamics, is what a
    # dead conditioning path looks like, and both are cheap to rule out.
    assert video.std() > 1.0, "output frames are nearly constant"
    per_frame = video.reshape(NUM_FRAMES, -1).mean(axis=1)
    assert per_frame.std() > 0.1, "every output frame has the same mean; the rollout may be static"


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_two_requests_are_served_independently():
    """Two identical requests must come back as two complete, identical videos.

    Not a batching test in disguise -- it is the *only* engine-level check that the two requests are
    not fused. The DiT shares one rotary position table and one KV cache scope across a batch, so a
    fused pair would generate sample 1 against sample 0's positions. ``forward()`` raises on batch > 1
    (see ``tests/diffusion/models/joyai_video_edit/test_pipeline_helpers.py``), but that guard is
    unreachable from here: ``_max_num_seqs`` defaults to 1 engine-wide, so through the engine the
    requests are serialised and both are valid. Asserting the raise here would be asserting a false
    premise -- what actually needs guarding is that the second request gets its own KV window and its
    own noise draw rather than inheriting the first one's.

    Equality is the assertion because the seed is fixed and the source is identical: same shapes, same
    device, same process, so the rollout is reproducible. A KV window that leaked across requests, or a
    noise generator seeded from global state instead of ``seed``, both break this and neither breaks
    the single-request test.
    """
    model_path = _require_model_path()
    source = _source_clip()
    request = {"prompt": PROMPT, "multi_modal_data": {"video": source}}

    with OmniRunner(model_path, model_class_name=MODEL_CLASS) as runner:
        outputs = runner.omni.generate([request, request], _sampling_params())

    assert len(outputs) == 2, f"expected two results, got {len(outputs)}"
    first, second = (output.images[0] for output in outputs)
    for video in (first, second):
        assert video.shape == (NUM_FRAMES, HEIGHT, WIDTH, 3), video.shape
    np.testing.assert_array_equal(first, second)
