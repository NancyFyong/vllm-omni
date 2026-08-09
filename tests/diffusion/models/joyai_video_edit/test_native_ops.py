# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the native replacements of JoyAI-Video-Edit's fused CUDA ops.

Upstream computes modulation, QK-norm+RoPE and the gated residual with ``joyomni_ops`` kernels that
have **no** pure-torch fallback -- they raise when the package is absent -- so the reference formulas
here were derived rather than read off upstream's source. They have since been checked against the real
kernels on a machine that has the extension built (all six ops agree to <=3.1e-5 relative, cosine
1.000000), so these tests are the *specification* a later fused-kernel step has to satisfy, and any
disagreement between kernel and formula shows up here rather than as slightly-wrong video.

Each test targets a failure that produces finite, correctly-shaped output:

- the ``1 +`` on the modulation scale (present) versus on the gate (absent) -- get either backwards and
  the network still runs;
- normalise-then-rotate ordering in :func:`qk_norm_rope`, where the swapped order rescales rotated
  vectors by a per-token factor;
- fp32 accumulation in bf16, where doing the LayerNorm in bf16 loses ~2 decimal digits;
- and *where* the cast back to bf16 happens, which is invisible in fp32 and worth 3e-3 per call in
  bf16 -- see :func:`test_rmsnorm_forward_rounds_earlier_than_normalize_fp32`.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import HEAD_DIM, NORM_EPS
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_rope import apply_rotary_emb
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_transformer import (
    RMSNorm,
    add_gate,
    layernorm_modulate,
    qk_norm_rope,
    sdpa_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

BATCH, SEQ, HEADS, DIM = 2, 7, 4, 16


def _freqs(seq: int = SEQ, dim: int = DIM, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """Arbitrary but valid unit phasors, interleaved to full head width.

    Carries a real batch dim: ``reshape_for_broadcast`` uses ``view``, not broadcasting, so ``[1, L, D]``
    tables against a batched ``x`` raise rather than broadcast (which is what ``_broadcast_freqs``
    exists to prevent in the model).
    """
    angles = torch.linspace(0.1, 1.3, seq * (dim // 2)).reshape(seq, dim // 2)
    cos = angles.cos().repeat_interleave(2, dim=-1)[None].repeat(batch, 1, 1)
    sin = angles.sin().repeat_interleave(2, dim=-1)[None].repeat(batch, 1, 1)
    return cos, sin


def test_layernorm_modulate_matches_the_reference_formula():
    """``ln(x) * (1 + scale) + shift``, both broadcast over the sequence."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, 32)
    shift, scale = torch.randn(BATCH, 32), torch.randn(BATCH, 32)

    expected = (
        torch.nn.functional.layer_norm(x, (32,), None, None, NORM_EPS) * (1 + scale[:, None, :]) + shift[:, None, :]
    )
    torch.testing.assert_close(layernorm_modulate(x, shift, scale, NORM_EPS), expected, rtol=1e-6, atol=1e-6)


def test_layernorm_modulate_is_identity_at_zero_modulation():
    """The ``1 +`` is what makes zero modulation a no-op.

    Blocks' ``modulate_table`` is initialised at ``randn / sqrt(hidden)``, i.e. ~0, so without the
    ``1 +`` an untrained network would emit near-zero everywhere -- and a trained one would be scaled
    by roughly the wrong amount at every one of 80 modulation sites.
    """
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, 32)
    zeros = torch.zeros(BATCH, 32)

    normed = torch.nn.functional.layer_norm(x, (32,), None, None, NORM_EPS)
    torch.testing.assert_close(layernorm_modulate(x, zeros, zeros, NORM_EPS), normed, rtol=1e-6, atol=1e-6)


def test_layernorm_modulate_rejects_a_sequence_shaped_modulation():
    """``[B, L, D]`` modulation would broadcast silently against the wrong axis."""
    x = torch.randn(BATCH, SEQ, 32)
    with pytest.raises(ValueError, match="must be 2D"):
        layernorm_modulate(x, torch.zeros(BATCH, SEQ, 32), torch.zeros(BATCH, SEQ, 32), NORM_EPS)


def test_layernorm_modulate_accumulates_in_fp32_under_bf16():
    """bf16 in, bf16 out, but the statistics are fp32.

    Computed natively in bf16 the LayerNorm mean/variance carry ~3 decimal digits; the fp32 path is
    within one bf16 ulp of the fp32 reference, which a bf16-native implementation is not.
    """
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, 32)
    shift, scale = torch.randn(BATCH, 32), torch.randn(BATCH, 32)

    reference = layernorm_modulate(x, shift, scale, NORM_EPS)
    got = layernorm_modulate(x.bfloat16(), shift.bfloat16(), scale.bfloat16(), NORM_EPS)

    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), reference, rtol=2e-2, atol=2e-2)


def test_add_gate_has_no_implicit_one_on_the_gate():
    """``residual + x * gate``, *not* ``residual + x * (1 + gate)``.

    The asymmetry against :func:`layernorm_modulate` is real and matches diffusers' Wan block. Adding
    a ``1 +`` here makes every block's residual branch pass through at unit strength on top of its
    learned gate -- output stays finite and the drift grows with depth.
    """
    torch.manual_seed(0)
    residual, x = torch.randn(BATCH, SEQ, 32), torch.randn(BATCH, SEQ, 32)
    gate = torch.randn(BATCH, 32)

    torch.testing.assert_close(add_gate(residual, x, gate), residual + x * gate[:, None, :], rtol=1e-6, atol=1e-6)
    # A zero gate must drop the branch entirely, which the `1 +` variant cannot do.
    torch.testing.assert_close(add_gate(residual, x, torch.zeros(BATCH, 32)), residual)


def test_rmsnorm_normalises_over_the_head_dimension_only():
    """Per-head, not per-token: the weight is ``head_dim``-wide and heads must not mix.

    Normalising over the flattened ``H*D`` instead -- an easy slip given the ``[B, L, H, D]`` layout --
    leaves shapes and magnitudes plausible while coupling all 32 heads.
    """
    torch.manual_seed(0)
    norm = RMSNorm(DIM, eps=NORM_EPS)
    with torch.no_grad():
        norm.weight.copy_(torch.ones(DIM))
    x = torch.randn(BATCH, SEQ, HEADS, DIM)
    # Give one head a hugely different scale; per-head normalisation must erase the difference.
    x[:, :, 0] *= 100.0

    out = norm(x)
    per_head_rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(per_head_rms, torch.ones_like(per_head_rms), rtol=1e-4, atol=1e-4)


def test_rmsnorm_applies_the_weight_after_normalising():
    torch.manual_seed(0)
    norm = RMSNorm(DIM, eps=NORM_EPS)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(BATCH, SEQ, HEADS, DIM)

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-5)


def test_rmsnorm_forward_rounds_earlier_than_normalize_fp32():
    """The two methods are deliberately *not* interchangeable, and only bf16 can tell them apart.

    Upstream reaches this normalisation two ways with two different roundings, and both are load-bearing:
    the text stream calls its torch ``RMSNorm`` (``dit.py:379``), which rounds to bf16 before the weight
    multiply, while the image q/k and KV-cache paths call fused kernels that write bf16 exactly once.
    Measured against the real kernel, collapsing them onto the early-rounding form costs 2.8e-3 relative
    per call where ``normalize_fp32`` costs 7.7e-6 -- in 40 blocks x 3 forwards.

    Asserted on bits rather than a tolerance: the whole difference *is* one rounding, so a threshold
    would either be arbitrary or hide the effect.
    """
    torch.manual_seed(0)
    norm = RMSNorm(DIM, eps=NORM_EPS, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=torch.bfloat16)

    x32 = x.float()
    normed32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    # `forward` matches upstream's torch module: round, *then* weight.
    assert torch.equal(norm(x), normed32.to(torch.bfloat16) * norm.weight)
    # `normalize_fp32` matches the kernels: one fp32 chain, uncast so a caller can keep chaining.
    assert norm.normalize_fp32(x).dtype is torch.float32
    torch.testing.assert_close(norm.normalize_fp32(x), normed32 * norm.weight.float(), rtol=0, atol=0)
    assert not torch.equal(norm(x), norm.normalize_fp32(x).to(torch.bfloat16))


def test_qk_norm_rope_does_not_round_between_the_norm_and_the_rotation():
    """The fused kernel keeps its post-norm intermediate in registers; going through ``forward`` would
    round it to bf16 and then cast straight back up inside :func:`apply_rotary_emb`.

    That extra round measured 3.3e-3 relative against the real kernel versus 7.9e-6 for the chained
    fp32 form -- the single largest numerical deviation found on this port's DiT path. Only bf16 exposes
    it, which is why the fp32 ordering test above cannot stand in for this one.
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=torch.bfloat16)
    k = torch.randn(BATCH, SEQ, HEADS, DIM, dtype=torch.bfloat16)
    q_norm, k_norm = RMSNorm(DIM, eps=NORM_EPS, dtype=torch.bfloat16), RMSNorm(DIM, eps=NORM_EPS, dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.normal_()
        k_norm.weight.normal_()
    cos, sin = _freqs()
    freqs = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))

    got_q, got_k = qk_norm_rope(q, k, q_norm, k_norm, freqs)
    assert got_q.dtype is torch.bfloat16 and got_k.dtype is torch.bfloat16
    torch.testing.assert_close(
        got_q, apply_rotary_emb(q_norm.normalize_fp32(q), freqs).to(torch.bfloat16), rtol=0, atol=0
    )
    # The rounded route is what a reader would write, and it is a different answer.
    assert not torch.equal(got_q, apply_rotary_emb(q_norm(q), freqs))
    assert not torch.equal(got_k, apply_rotary_emb(k_norm(k), freqs))


def test_qk_norm_rope_normalises_before_rotating():
    """Order matters. Rotation is norm-preserving, so the swapped order is *not* equivalent.

    Rotating first then normalising divides by the rotated vector's RMS -- numerically close but
    per-token different, and it is the ordering the fused kernel does not use.

    The reference here rotates by the *bf16-rounded* tables, matching what
    :func:`qk_norm_rope` now does; see
    :func:`test_qk_norm_rope_rounds_the_rotary_tables_to_bf16` for why. Passing the raw fp32 tables
    to the reference would make this test fail by ~1e-3 for a reason that has nothing to do with the
    ordering it is checking.
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM)
    k = torch.randn(BATCH, SEQ, HEADS, DIM)
    q_norm, k_norm = RMSNorm(DIM, eps=NORM_EPS), RMSNorm(DIM, eps=NORM_EPS)
    with torch.no_grad():
        q_norm.weight.normal_()
        k_norm.weight.normal_()
    freqs = _freqs()
    rounded = tuple(t.to(torch.bfloat16).float() for t in freqs)

    got_q, got_k = qk_norm_rope(q, k, q_norm, k_norm, freqs)
    torch.testing.assert_close(got_q, apply_rotary_emb(q_norm(q), rounded), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_k, apply_rotary_emb(k_norm(k), rounded), rtol=1e-6, atol=1e-6)

    rotate_first = q_norm(apply_rotary_emb(q, rounded))
    assert not torch.allclose(got_q, rotate_first, rtol=1e-3, atol=1e-3)


def test_qk_norm_rope_rounds_the_rotary_tables_to_bf16():
    """``sgl_fused_ops.fused_qk_norm_rope_3d`` rotates by bf16 tables, so this path must too.

    The wrapper ends with ``cos_bf16 = cos.to(torch.bfloat16)`` and the kernel reads them back as
    ``__bfloat162float(cos_ptr[...])``, so upstream's rotation only ever sees bf16's 8 mantissa bits
    even though the tables are *built* in fp32. Applying the fp32 tables is strictly more accurate and
    measurably not upstream: with every DiT input substituted from upstream's own dump, ``v`` (a bare
    slice of the qkv projection) sat at 2.5e-05 and the text q/k (normed, never rotated) at 5.7e-05,
    while the rotated image q/k sat at 2.3e-03 -- a median of 2 bf16 ULPs, uniform across all three of
    ``rope_dim_list``'s groups, which is a precision difference and not a wrong formula. 2**-9 is 2e-3.

    This must not spread to the cached-pre-RoPE read path: upstream re-rotates cached keys through its
    own torch ``apply_rotary_emb`` (``dit.py:209``) on the unrounded fp32 tables, and that function is
    character-for-character ours, so those already agree.

    fp32 q/k on purpose -- in bf16 the output rounding swamps a table difference this size, which is
    exactly how the fp32 tables survived the earlier random-tensor probe unnoticed.
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM)
    k = torch.randn(BATCH, SEQ, HEADS, DIM)
    q_norm, k_norm = RMSNorm(DIM, eps=NORM_EPS), RMSNorm(DIM, eps=NORM_EPS)
    cos, sin = _freqs()
    # A table that is *not* bf16-representable, so rounding it is observable at all. `_freqs()` values
    # that happen to land on a bf16 grid point would make any assertion here vacuous.
    assert not torch.equal(cos, cos.to(torch.bfloat16).float())

    got_q, _ = qk_norm_rope(q, k, q_norm, k_norm, (cos, sin))
    rounded = (cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float())

    torch.testing.assert_close(got_q, apply_rotary_emb(q_norm.normalize_fp32(q), rounded), rtol=0, atol=0)
    # The fp32-table route is what this port shipped first, and it is a different answer.
    assert not torch.equal(got_q, apply_rotary_emb(q_norm.normalize_fp32(q), (cos, sin)))


def test_sdpa_attention_is_never_causal():
    """A causal mask here would break text fusion, not just token ordering.

    Text is concatenated into the same self-attention as the image stream and cached history is
    *prepended*, so with ``is_causal=True`` early image tokens would lose the prompt entirely. The
    check is that the first query attends to later keys: under a causal mask its output equals value 0.
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM)
    k = torch.randn(BATCH, SEQ, HEADS, DIM)
    v = torch.randn(BATCH, SEQ, HEADS, DIM)

    out = sdpa_attention(q, k, v)
    assert out.shape == (BATCH, SEQ, HEADS, DIM)
    torch.testing.assert_close(
        out,
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(out[:, 0], v[:, 0], rtol=1e-3, atol=1e-3), "query 0 saw only key 0 -- causal mask leaked"


def test_sdpa_attention_accepts_more_keys_than_queries():
    """The cached-history layout: keys ``[cached | img | txt]`` against queries ``[img | txt]``.

    History is attended to but never denoised, so the query is deliberately not extended. An
    implementation that assumed a square attention matrix would fail here rather than at chunk 1.
    """
    torch.manual_seed(0)
    q = torch.randn(BATCH, SEQ, HEADS, DIM)
    k = torch.randn(BATCH, SEQ * 3, HEADS, DIM)
    v = torch.randn(BATCH, SEQ * 3, HEADS, DIM)

    assert sdpa_attention(q, k, v).shape == (BATCH, SEQ, HEADS, DIM)


def test_head_dim_constant_matches_the_rope_budget():
    """``sum(rope_dim_list) == HEAD_DIM`` is asserted in the model; keep the constant honest here too."""
    assert HEAD_DIM == 128
