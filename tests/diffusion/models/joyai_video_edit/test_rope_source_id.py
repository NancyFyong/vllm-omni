# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JoyAI-Video-Edit rotary embeddings.

Each test here guards a port failure that shape-only checks pass straight through:

- The interleaved (GPT-J) pair layout can be swapped for the half-split (NeoX) one without any
  shape change, and on a square grid a transposed ``(h, w)`` is also invisible -- so the reference
  comparison uses an independent complex-multiply implementation on a *non-square* grid.
- ``compose_rope`` folding two rotations into one is a closed-form rotation identity, checkable to
  fp64 machine epsilon. A wrong sign or operand order still produces plausible-looking video.
- Source-id RoPE is the *only* signal separating the noisy target chunk from the clean source chunk
  (both are given identical ``(t, h, w)`` positions). A port that drops it entirely still produces
  correctly-shaped, finite output that quietly ignores the edit condition.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit import joyai_video_edit_rope as rope
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    HEAD_DIM,
    ROPE_DIM_LIST,
    SOURCE_ID_EDIT_CONDITION,
    SOURCE_ID_EXTRA_REF_IMAGE,
    SOURCE_ID_TARGET,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Deliberately non-square and multi-frame: h == w hides transposed spatial axes.
T, H, W = 2, 3, 5
NUM_TOKENS = T * H * W
CPU = torch.device("cpu")


def _reference_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE via complex multiplication, independently of the implementation.

    Shares no code with ``apply_rotary_emb`` -- in particular it does not use ``rotate_half`` -- so a
    sign error or a half-split/interleaved mix-up in the port cannot cancel out.
    """
    z = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2).contiguous())
    phase = torch.view_as_complex(torch.stack([cos.double()[..., ::2], sin.double()[..., ::2]], dim=-1).contiguous())
    return torch.view_as_real(z * phase.unsqueeze(-2)).flatten(-2)


@pytest.fixture
def freqs_3d() -> tuple[torch.Tensor, torch.Tensor]:
    frame_ids = rope.get_token_frame_ids((T, H, W), CPU)
    return rope.get_rotary_pos_embed_from_ids(frame_ids=frame_ids, spatial_shape=(H, W), head_dim=HEAD_DIM)


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, NUM_TOKENS, 2, HEAD_DIM, dtype=torch.float64)


def test_rope_dims_sum_to_head_dim():
    """The runtime asserts this deep inside the DiT; catch a bad config here instead."""
    assert sum(ROPE_DIM_LIST) == HEAD_DIM


def test_token_frame_ids_are_frame_major():
    """Token order must match ``img_in(x).flatten(2).transpose(1, 2)``.

    Using ``repeat`` instead of ``repeat_interleave`` yields the same length but interleaves frames,
    silently scrambling temporal positions.
    """
    assert rope.get_token_frame_ids((T, H, W), CPU).tolist() == [0] * (H * W) + [1] * (H * W)


def test_token_frame_ids_honour_explicit_temporal_ids():
    """Renumbered window positions must survive to the RoPE tables."""
    ids = torch.tensor([0, 2])
    assert rope.get_token_frame_ids((T, H, W), CPU, temporal_ids=ids).tolist() == [0] * (H * W) + [2] * (H * W)


def test_token_frame_ids_reject_wrong_length():
    with pytest.raises(ValueError, match="must be 1D with length"):
        rope.get_token_frame_ids((T, H, W), CPU, temporal_ids=torch.tensor([0, 1, 2]))


def test_rotary_pos_embed_rejects_mismatched_rope_dims(freqs_3d):
    """``sum(rope_dim_list) != head_dim`` must raise rather than silently produce a short table."""
    frame_ids = rope.get_token_frame_ids((T, H, W), CPU)
    with pytest.raises(ValueError, match="must equal head_dim"):
        rope.get_rotary_pos_embed_from_ids(
            frame_ids=frame_ids, spatial_shape=(H, W), head_dim=HEAD_DIM, rope_dim_list=[16, 56, 8]
        )


def test_interleaved_layout_matches_complex_reference(freqs_3d, tokens):
    """Interleaved pair layout, verified on a non-square grid.

    Tolerance is fp32-level because ``apply_rotary_emb`` computes in fp32 (as upstream does), not
    because the layout is approximate.
    """
    got = rope.apply_rotary_emb(tokens, freqs_3d)
    expected = _reference_apply(tokens, *freqs_3d)
    torch.testing.assert_close(got.double(), expected, atol=1e-6, rtol=1e-6)


def test_transposing_spatial_axes_changes_the_tables():
    """Guards against feeding ``(w, h)`` where ``(h, w)`` is meant -- undetectable on a square grid."""
    normal = rope.get_rotary_pos_embed_from_ids(
        frame_ids=rope.get_token_frame_ids((T, H, W), CPU), spatial_shape=(H, W), head_dim=HEAD_DIM
    )
    swapped = rope.get_rotary_pos_embed_from_ids(
        frame_ids=rope.get_token_frame_ids((T, W, H), CPU), spatial_shape=(W, H), head_dim=HEAD_DIM
    )
    assert not torch.allclose(normal[0], swapped[0])


def test_apply_rotary_emb_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even head dimension"):
        rope.apply_rotary_emb(torch.randn(1, 4, 2, 5), (torch.ones(4, 5), torch.zeros(4, 5)))


def test_compose_rope_equals_sequential_rotation(freqs_3d, tokens):
    """Rotating by the composed angle == rotating by each in turn, to fp64 machine epsilon.

    This is a closed-form property of rotations, so it pins down the sign convention and operand
    order exactly -- the two things a plausible-but-wrong composition gets backwards.
    """
    source_id = torch.full((1, NUM_TOKENS), SOURCE_ID_EDIT_CONDITION)
    freqs_role = rope.generate_source_id_rope(source_id, HEAD_DIM, CPU, torch.float64)
    cos_3d, sin_3d = (f.unsqueeze(0).double() for f in freqs_3d)

    sequential = _reference_apply(_reference_apply(tokens, cos_3d, sin_3d), *freqs_role)
    composed = _reference_apply(tokens, *rope.compose_rope((cos_3d, sin_3d), freqs_role))

    torch.testing.assert_close(sequential, composed, atol=1e-12, rtol=1e-12)


def test_source_id_zero_is_exactly_identity():
    """Target tokens must be left untouched by the role stage, exactly -- not just approximately."""
    cos, sin = rope.generate_source_id_rope(torch.zeros(1, 4), HEAD_DIM, CPU, torch.float64)
    assert torch.equal(cos, torch.ones_like(cos))
    assert torch.equal(sin, torch.zeros_like(sin))


@pytest.mark.parametrize("other_id", [SOURCE_ID_EDIT_CONDITION, SOURCE_ID_EXTRA_REF_IMAGE])
def test_source_id_changes_q_at_identical_positions(freqs_3d, tokens, other_id):
    """The load-bearing assertion.

    The source video latent is given the *same* temporal ids as the noisy chunk, so these tokens are
    at identical ``(t, h, w)``. If ``source_id`` does not change the result, the model cannot
    distinguish the frame it is editing from the frame it is denoising, and the edit condition is
    invisible -- while every shape and finiteness check still passes.
    """
    cos_3d, sin_3d = (f.unsqueeze(0).float() for f in freqs_3d)

    def rotated(source_id_value: float) -> torch.Tensor:
        source_id = torch.full((1, NUM_TOKENS), source_id_value)
        freqs_role = rope.generate_source_id_rope(source_id, HEAD_DIM, CPU, torch.float32)
        return rope.apply_rotary_emb(tokens.float(), rope.compose_rope((cos_3d, sin_3d), freqs_role))

    assert (rotated(SOURCE_ID_TARGET) - rotated(other_id)).abs().max() > 1e-3
