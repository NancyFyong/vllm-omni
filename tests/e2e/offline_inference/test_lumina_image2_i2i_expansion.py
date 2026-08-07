# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for Lumina-Image-2.0 image-to-image (SDEdit) generation.

The same text-to-image checkpoint is served; passing an image via
``multi_modal_data.image`` triggers the SDEdit path, and ``strength`` selects
how far back into the schedule the input image is re-noised.
"""

import pytest
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "Alpha-VLLM/Lumina-Image-2.0"

_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "parallel_config": DiffusionParallelConfig(
            tensor_parallel_size=2,
        ),
        "enable_cpu_offload": True,
    },
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_lumina_image2_image_to_image(omni_runner_handler: OmniRunnerHandler) -> None:
    input_image = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
    request_config = {
        "model": omni_runner_handler.runner.model_name,
        "prompt": "A watercolour painting, soft pastel colours",
        "multi_modal_data": {"image": input_image},
        "sampling_params": OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=2,
            guidance_scale=4.0,
            strength=0.6,
            seed=42,
        ),
    }
    omni_runner_handler.send_diffusion_request(request_config)
