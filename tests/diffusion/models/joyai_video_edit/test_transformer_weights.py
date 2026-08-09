# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight checks for the JoyAI-Video-Edit DiT.

Skipped unless ``JOYAI_VIDEO_EDIT_MODEL_DIR`` points at a local checkout of the weights, so CI stays
green without the 30 GiB checkpoint. Two things need it:

**The strict load itself is the assertion.** Both key lists empty means all 16,263,675,968 parameters
landed in correctly-named slots, i.e. the module tree matches the checkpoint's. That is most of what
can be verified without a side-by-side reference, and it is the port's single highest-risk step: the
checkpoint's key names encode diffusers internals (``img_mlp.net.0.proj.weight``,
``linear_1``/``linear_2``), so a diffusers refactor breaks it. ``test_transformer.py`` pins the same key
names on CPU in milliseconds; this confirms the pinning against the real file.

**Output scale at production resolution.** A flow-matching velocity prediction on a
unit-variance latent should itself be roughly unit-variance. Getting the modulation formula wrong --
a missing ``1 +`` on the scale, or a stray one on the gate -- leaves shapes and finiteness intact but
moves this number, and it is the cheapest available check on the 80 modulation sites.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MAX_CONDITION_TOKENS,
    TEXT_STATES_DIM,
    TOTAL_DIT_PARAMS,
    VAE_LATENT_CHANNELS,
    VAE_SPATIAL_COMPRESSION,
    dit_config,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_kv import JoyKVWindow
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_transformer import (
    JoyAIVideoEditTransformer3DModel,
)

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.gpu]

MODEL_DIR_ENV = "JOYAI_VIDEO_EDIT_MODEL_DIR"
CHECKPOINT = "dit/joyai_video_edit_dit_0804.pth"
GRID_H = DEFAULT_HEIGHT // VAE_SPATIAL_COMPRESSION  # 30
GRID_W = DEFAULT_WIDTH // VAE_SPATIAL_COMPRESSION  # 52


def _checkpoint_path() -> Path:
    root = os.environ.get(MODEL_DIR_ENV)
    if not root:
        pytest.skip(f"set {MODEL_DIR_ENV} to the JoyAI-Video-Edit weights directory to run this")
    path = Path(root) / CHECKPOINT
    if not path.is_file():
        pytest.skip(f"{path} does not exist")
    if not torch.accelerator.is_available():
        pytest.skip("the 16.2B DiT needs an accelerator; a CPU forward is impractically slow")
    return path


@pytest.fixture(scope="module")
def loaded_dit() -> JoyAIVideoEditTransformer3DModel:
    """Load the production way: ``meta`` construction, then ``mmap`` + ``assign``.

    Without ``mmap=True`` ``torch.load`` materialises all 30 GiB in host RAM, and without
    ``assign=True`` ``load_state_dict`` copies into separately-allocated parameters for a second 30 GiB.
    Together they make this a near-zero-copy read, which is why it takes seconds rather than minutes.
    """
    path = _checkpoint_path()
    model = JoyAIVideoEditTransformer3DModel(**dit_config(), device="meta", dtype=torch.bfloat16)
    state_dict = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    result = model.load_state_dict(state_dict, strict=True, assign=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    del state_dict
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def _noise(frames: int = 1) -> torch.Tensor:
    return torch.randn(1, VAE_LATENT_CHANNELS, frames, GRID_H, GRID_W, device="cuda", dtype=torch.bfloat16)


def test_the_real_checkpoint_loads_strictly(loaded_dit):
    """Every parameter accounted for, and none of them left on ``meta``.

    ``assign=True`` replaces parameters wholesale, so a key the checkpoint does not cover would stay a
    meta tensor and fail at the first matmul with an unhelpful error instead of here.
    """
    assert sum(p.numel() for p in loaded_dit.parameters()) == TOTAL_DIT_PARAMS
    assert {p.dtype for p in loaded_dit.parameters()} == {torch.bfloat16}
    assert not [name for name, p in loaded_dit.named_parameters() if p.is_meta]


def test_velocity_prediction_is_unit_scale_at_production_resolution(loaded_dit):
    """720x1248 -> a 30x52 latent grid, and a velocity prediction of about unit variance.

    The bounds are wide enough to be dtype- and seed-insensitive but far tighter than "finite": a
    modulation or gating formula error moves the standard deviation by a large factor while leaving
    every shape and finiteness check green.
    """
    torch.manual_seed(0)
    window = JoyKVWindow()
    with torch.no_grad():
        out, _ = loaded_dit(
            _noise(),
            torch.full((1,), 1000.0, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, MAX_CONDITION_TOKENS, TEXT_STATES_DIM, device="cuda", dtype=torch.bfloat16),
            ref_video_latent=_noise(),
            current_temporal_ids=torch.tensor([[0]], device="cuda"),
            kv_window=window,
            kv_cache_mode="reuse",
            kv_cache_scope="cond",
            kv_cache_chunk_id=0,
            kv_cache_selected_chunk_ids=[],
            kv_cache_pre_rope=True,
        )

    assert tuple(out.shape) == (1, VAE_LATENT_CHANNELS, 1, GRID_H, GRID_W)
    assert torch.isfinite(out).all()
    std = out.float().std().item()
    assert 0.5 < std < 2.0, f"velocity std {std:.3f} is not unit-scale"


def test_cached_chunk_costs_the_predicted_amount_of_memory(loaded_dit):
    """One chunk of KV is ``tokens x heads x head_dim x 2 x layers x 2 bytes``, and no more.

    1560 tokens over 40 layers is ~1.0 GiB, which is what bounds the window to three chunks. Caching
    the text stream as well -- easy to do, since text is concatenated into the same attention -- would
    inflate this by two thirds at 1024 condition tokens while breaking nothing visibly.
    """
    torch.manual_seed(0)
    window = JoyKVWindow()
    torch.accelerator.empty_cache()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        loaded_dit(
            _noise(),
            torch.zeros(1, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, MAX_CONDITION_TOKENS, TEXT_STATES_DIM, device="cuda", dtype=torch.bfloat16),
            current_temporal_ids=torch.tensor([[0]], device="cuda"),
            kv_window=window,
            kv_cache_mode="store",
            kv_cache_scope="cond",
            kv_cache_chunk_id=0,
            kv_cache_selected_chunk_ids=[],
            kv_cache_pre_rope=True,
            skip_text_stream=True,
        )
    resident_bytes = torch.cuda.memory_allocated() - before

    expected = GRID_H * GRID_W * loaded_dit.hidden_size * 2 * len(loaded_dit.double_blocks) * 2
    assert window.resident_chunk_ids("cond") == {0}
    # Generous upper bound: exact equality would be hostage to allocator rounding and stray temporaries.
    assert resident_bytes < expected * 1.5, f"cached {resident_bytes / 2**30:.2f} GiB, expected ~{expected / 2**30:.2f}"


def test_reusing_the_stored_chunk_changes_the_prediction(loaded_dit):
    """End-to-end confirmation that the pre-RoPE cache path is live under real weights.

    The CPU tests show the plumbing is connected; this shows the *rotated* cached keys are still in
    range for a trained model -- a wrong rotation would typically saturate attention and collapse the
    output's variance rather than raise.
    """
    torch.manual_seed(0)
    text = torch.randn(1, MAX_CONDITION_TOKENS, TEXT_STATES_DIM, device="cuda", dtype=torch.bfloat16)
    current, reference = _noise(), _noise()
    window = JoyKVWindow()

    with torch.no_grad():
        loaded_dit(
            _noise(),
            torch.zeros(1, device="cuda", dtype=torch.bfloat16),
            text,
            current_temporal_ids=torch.tensor([[0]], device="cuda"),
            kv_window=window,
            kv_cache_mode="store",
            kv_cache_scope="cond",
            kv_cache_chunk_id=0,
            kv_cache_selected_chunk_ids=[],
            kv_cache_pre_rope=True,
            skip_text_stream=True,
        )
        denoise = {
            "timestep": torch.full((1,), 837.636, device="cuda", dtype=torch.bfloat16),
            "encoder_hidden_states": text,
            "ref_video_latent": reference,
            "current_temporal_ids": torch.tensor([[1]], device="cuda"),
            "kv_cache_pre_rope": True,
        }
        with_cache, _ = loaded_dit(
            current,
            **denoise,
            cached_temporal_ids=torch.tensor([[0]], device="cuda"),
            kv_window=window,
            kv_cache_mode="reuse",
            kv_cache_scope="cond",
            kv_cache_chunk_id=1,
            kv_cache_selected_chunk_ids=[0],
        )
        without_cache, _ = loaded_dit(current, **denoise)

    assert torch.isfinite(with_cache).all()
    assert not torch.allclose(with_cache, without_cache, rtol=1e-2, atol=1e-2)
    std = with_cache.float().std().item()
    assert 0.5 < std < 2.0, f"velocity std {std:.3f} with history is not unit-scale"
