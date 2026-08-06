"""
Recommended tests of diffusion features that are available in online serving mode
and are supported by the following model:
- Lumina-Image-2.0 (Next-DiT): text-to-image with single prompt input
Coverage:
- Default smoke (1 GPU)
- CPU offloading (model-level sequential offload via --enable-cpu-offload)
- Cache-DiT acceleration (1 GPU)
- CFG-Parallel (2 GPU)

This validates:
 - Successful image generation at the expected 1024x1024 resolution with recommended
   feature combinations
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

TEXT_TO_IMAGE_PROMPT = "A serene mountain lake at sunset, photorealistic, highly detailed."
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

MODEL = "Alpha-VLLM/Lumina-Image-2.0"


def _get_diffusion_feature_cases(model: str):
    """Return diffusion feature cases for Lumina-Image-2.0."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--enable-cpu-offload"],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--cache-backend", "cache_dit"],
            ),
            id="single_card_002",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_lumina_image2(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test the recommended feature combinations for Lumina-Image-2.0."""
    messages = dummy_messages_from_mix_data(content_text=TEXT_TO_IMAGE_PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 2,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
