# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lumina-Image-2.0 (Next-DiT) native integration for vLLM-Omni."""

from vllm_omni.diffusion.models.lumina_image2.lumina2_transformer import (
    Lumina2Transformer2DModel,
)
from vllm_omni.diffusion.models.lumina_image2.pipeline_lumina2 import (
    Lumina2Pipeline,
    get_lumina2_post_process_func,
)

__all__ = [
    "Lumina2Pipeline",
    "Lumina2Transformer2DModel",
    "get_lumina2_post_process_func",
]
