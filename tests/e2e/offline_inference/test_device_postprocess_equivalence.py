# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end proof that reducing a video on the device changes no pixels.

The per-model unit tests compare the device reduction against each pipeline's own
postprocessor on a captured tensor. This runs the whole engine twice with the same
seed instead, once on the float path and once on the reduced path, and requires
the frames a client would receive to be byte-identical.

Point it at any of the six supported video pipelines::

    VLLM_OMNI_DEVICE_POSTPROCESS_MODEL=robbyant/lingbot-video-dense-1.3b \
        pytest -s tests/e2e/offline_inference/test_device_postprocess_equivalence.py

Measured on lingbot-video-dense-1.3b at 17x256x256: 0 of 3342336 values differ.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_MODEL")

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODEL,
        reason="Set VLLM_OMNI_DEVICE_POSTPROCESS_MODEL to a video pipeline to run this.",
    ),
]

_SEED = 1234
_STEPS = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_STEPS", "2"))
_SIZE = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_SIZE", "256"))
_FRAMES = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_FRAMES", "17"))


def _generate(enable_device_postprocess: bool) -> np.ndarray:
    from vllm_omni.entrypoints.omni import Omni

    engine = Omni(
        model=MODEL,
        num_gpus=1,
        video_output_transport={"enable_device_postprocess": enable_device_postprocess},
    )
    try:
        # output_type must travel with the request: the pipelines read it from the
        # sampling params, and the reduction only covers the "np" path.
        sampling_params = OmniDiffusionSamplingParams(
            output_type="np",
            seed=_SEED,
            num_inference_steps=_STEPS,
            height=_SIZE,
            width=_SIZE,
            num_frames=_FRAMES,
        )
        outputs = engine.generate({"prompt": "a robot waving"}, sampling_params)
    finally:
        engine.close()

    video = outputs[0].images[0]
    return video.numpy() if hasattr(video, "numpy") else np.asarray(video)


def _as_uint8_frames(video: np.ndarray) -> np.ndarray:
    """Reproduce what the API server does before encoding.

    A float payload is scaled and rounded there, so that is the only fair place to
    compare it against a payload the worker already reduced.
    """
    if video.dtype == np.uint8:
        return video
    return np.rint(np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_device_postprocess_produces_identical_frames() -> None:
    float_path = _generate(False)
    device_path = _generate(True)

    # Guard against the run silently taking the float path twice, which would make
    # the comparison meaningless.
    assert float_path.dtype != np.uint8, "the float path returned uint8; the gate may be stuck open"
    assert device_path.dtype == np.uint8, (
        "the device path returned float; the reduction did not run. "
        "output_type must be 'np' and the model must be one of the supported pipelines."
    )
    assert float_path.shape == device_path.shape

    expected = _as_uint8_frames(float_path)
    np.testing.assert_array_equal(device_path, expected)
