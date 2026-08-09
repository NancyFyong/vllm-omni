# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the JoyAI-Video-Edit DiT, at a tiny config on CPU with no weights.

Three groups, each guarding something the real 30 GiB strict load cannot:

**Checkpoint-key pinning.** The shipped checkpoint's key names encode *diffusers'* internals
(``img_mlp.net.0.proj.weight``, ``linear_1``/``linear_2``), so this port imports ``FeedForward``,
``TimestepEmbedding`` and ``PixArtAlphaTextProjection`` rather than copying them. That makes a
diffusers refactor a real risk, and these tests turn it into a millisecond CPU failure naming the
offending key instead of a 30 GiB load that reports hundreds of missing keys.

**Wiring the module tree cannot express.** ``strict=True`` proves every parameter landed in a
correctly-named slot; it says nothing about whether the reference latent is concatenated, whether
cached keys are prepended, or whether text is excluded from the cache. Those are asserted here.

**Loud failure where upstream is silent.** Upstream takes ``current_temporal_ids[0]`` and discards the
rest of the batch, and accepts ``encoder_hidden_states_mask`` without ever reading it. Both are pinned
so a future reader cannot "fix" either into a different behaviour unnoticed.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    HIDDEN_SIZE,
    MM_DOUBLE_BLOCKS_DEPTH,
    NUM_MODULATION_CHUNKS,
    TIME_FREQ_DIM,
    TOTAL_DIT_PARAMS,
    dit_config,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_kv import JoyKVWindow
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_transformer import (
    JoyAIVideoEditTransformer3DModel,
    MMDoubleStreamBlock,
    ModulateWan,
    WanTimeTextImageEmbedding,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# 3 x 5 is deliberately non-square and prime-ish: a transposed spatial rearrange cannot survive it.
FRAMES, GRID_H, GRID_W = 1, 3, 5
TOKENS_PER_FRAME = GRID_H * GRID_W
TEXT_LEN = 7
TINY = {
    "hidden_size": 256,
    "heads_num": 2,
    "mm_double_blocks_depth": 2,
    "text_states_dim": 64,
    "rope_dim_list": [16, 56, 56],
}


@pytest.fixture
def model() -> JoyAIVideoEditTransformer3DModel:
    torch.manual_seed(0)
    return JoyAIVideoEditTransformer3DModel(**TINY, dtype=torch.float32).eval()


def _latent(batch: int = 1, frames: int = FRAMES) -> torch.Tensor:
    return torch.randn(batch, 64, frames, GRID_H, GRID_W)


def _inputs(batch: int = 1) -> dict:
    return {
        "hidden_states": _latent(batch),
        "timestep": torch.full((batch,), 1000.0),
        "encoder_hidden_states": torch.randn(batch, TEXT_LEN, TINY["text_states_dim"]),
    }


# --- checkpoint-key pinning ----------------------------------------------------------------------


def test_block_state_dict_keys_match_the_shipped_checkpoint_layout():
    """The 22 keys per block, verbatim from ``joyai_video_edit_dit_0804.pth``.

    ``net.0.proj``/``net.2`` come from diffusers' ``FeedForward`` with ``gelu-approximate`` (the GELU
    sits at ``net.0``, dropout at ``net.1``); a diffusers change to either the container name or the
    activation slot renames them and breaks the load.
    """
    block = MMDoubleStreamBlock(hidden_size=256, heads_num=2, mlp_width_ratio=4.0)
    expected = {
        f"{stream}_{suffix}"
        for stream in ("img", "txt")
        for suffix in (
            "mod.modulate_table",
            "attn_qkv.weight",
            "attn_qkv.bias",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_proj.weight",
            "attn_proj.bias",
            "mlp.net.0.proj.weight",
            "mlp.net.0.proj.bias",
            "mlp.net.2.weight",
            "mlp.net.2.bias",
        )
    }
    assert set(block.state_dict()) == expected


def test_norms_are_affine_free_and_contribute_no_keys():
    """``img_norm1``/``img_norm2`` supply only ``eps``; modulation supplies scale and shift.

    Constructing them with ``elementwise_affine=True`` would add four unexpected keys per block and
    fail the strict load -- but only after the full 30 GiB read.
    """
    block = MMDoubleStreamBlock(hidden_size=256, heads_num=2, mlp_width_ratio=4.0)
    for name in ("img_norm1", "img_norm2", "txt_norm1", "txt_norm2"):
        norm = getattr(block, name)
        assert norm.elementwise_affine is False
        assert list(norm.parameters()) == []


def test_condition_embedder_state_dict_keys_match_the_shipped_checkpoint_layout():
    """``linear_1``/``linear_2`` on both the time and text projections come from diffusers.

    ``timesteps_proj`` is parameter-free and ``act_fn``/``act_1`` hold no weights, so exactly six
    entries are expected.
    """
    embedder = WanTimeTextImageEmbedding(
        dim=256, time_freq_dim=TIME_FREQ_DIM, time_proj_dim=256 * NUM_MODULATION_CHUNKS, text_embed_dim=64
    )
    assert set(embedder.state_dict()) == {
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
        "time_proj.weight",
        "time_proj.bias",
        "text_embedder.linear_1.weight",
        "text_embedder.linear_1.bias",
        "text_embedder.linear_2.weight",
        "text_embedder.linear_2.bias",
    }


def test_real_config_construction_matches_the_checkpoint_parameter_count():
    """16,263,675,968 parameters, all bf16, built on ``meta`` so this stays a CPU test.

    The dtype half matters independently: upstream constructs ``img_in`` and ``condition_embedder``
    at the default fp32 and blanket-casts afterwards, which for this model transiently needs ~4x its
    final footprint. The factory context is what avoids that, and a regression in it is invisible
    except as an out-of-memory much later.
    """
    model = JoyAIVideoEditTransformer3DModel(**dit_config(), device="meta", dtype=torch.bfloat16)
    assert sum(p.numel() for p in model.parameters()) == TOTAL_DIT_PARAMS
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert len(model.state_dict()) == 14 + 22 * MM_DOUBLE_BLOCKS_DEPTH
    # Parameter-free head: the checkpoint has no `norm_out.*` keys.
    assert not any(k.startswith("norm_out") for k in model.state_dict())


def test_modulate_table_splits_into_six_shift_scale_gate_pairs():
    """AdaLN-single order: ``shift1, scale1, gate1, shift2, scale2, gate2``.

    Verified by construction rather than by value: chunking along the wrong axis, or with the wrong
    count, is the kind of error that leaves magnitudes plausible.
    """
    mod = ModulateWan(HIDDEN_SIZE, NUM_MODULATION_CHUNKS)
    assert mod.modulate_table.shape == (1, NUM_MODULATION_CHUNKS, HIDDEN_SIZE)
    parts = mod(torch.zeros(2, NUM_MODULATION_CHUNKS, HIDDEN_SIZE))
    assert len(parts) == NUM_MODULATION_CHUNKS
    assert all(p.shape == (2, HIDDEN_SIZE) for p in parts)
    # Each part must be its own slice of the table, not a shared view of the whole thing.
    for i, part in enumerate(parts):
        torch.testing.assert_close(part, mod.modulate_table[0, i].expand(2, -1))


# --- geometry and guards -------------------------------------------------------------------------


def test_output_is_a_latent_of_the_input_shape(model):
    torch.manual_seed(0)
    inputs = _inputs()
    with torch.no_grad():
        out, txt = model(**inputs)
    assert out.shape == inputs["hidden_states"].shape
    assert torch.isfinite(out).all()
    # The text stream is returned too, projected to hidden width.
    assert txt.shape == (1, TEXT_LEN, TINY["hidden_size"])


def test_reference_latent_is_dropped_from_the_output_not_decoded(model):
    """``ref_video_latent`` doubles the image stream inside the blocks but not the output.

    Reference tokens are *appended*, so the target is the leading slice. Returning the whole thing
    would produce a latent with twice the frames -- caught here, whereas returning the *trailing*
    slice would have the right shape and be wrong.
    """
    torch.manual_seed(0)
    inputs = _inputs()
    with torch.no_grad():
        without, _ = model(**inputs)
        with_ref, _ = model(**inputs, ref_video_latent=_latent())
    assert with_ref.shape == without.shape
    assert not torch.allclose(with_ref, without, rtol=1e-3, atol=1e-3), "reference latent had no effect"


def test_reference_latent_must_share_the_spatial_grid(model):
    """A mismatched reference grid would otherwise concatenate and shift every position."""
    torch.manual_seed(0)
    bad_ref = torch.randn(1, 64, FRAMES, GRID_H + 1, GRID_W)
    with pytest.raises(ValueError, match="spatial patch shape"):
        with torch.no_grad():
            model(**_inputs(), ref_video_latent=bad_ref)


def test_encoder_hidden_states_mask_is_provably_ignored(model):
    """Accepted for signature compatibility, never read.

    Text is attended to unmasked as part of the concatenated self-attention -- upstream casts this to
    bool and drops it. Two contradictory masks giving bit-identical output is the only way to state
    that as a fact rather than a comment.
    """
    torch.manual_seed(0)
    inputs = _inputs()
    with torch.no_grad():
        ones, _ = model(**inputs, encoder_hidden_states_mask=torch.ones(1, TEXT_LEN, dtype=torch.bool))
        zeros, _ = model(**inputs, encoder_hidden_states_mask=torch.zeros(1, TEXT_LEN, dtype=torch.bool))
    assert torch.equal(ones, zeros)


def test_batch_with_differing_positions_is_rejected_rather_than_silently_truncated(model):
    """One rotary table is shared by the batch, so per-sample positions cannot be honoured.

    Upstream indexes ``[0]`` and discards samples 1..N: shapes stay right and the extra samples come
    back denoised at the wrong positions. Raising is the earliest point that is catchable.
    """
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="differs across the batch"):
        with torch.no_grad():
            model(**_inputs(batch=2), current_temporal_ids=torch.tensor([[0], [1]]))


def test_batch_with_identical_positions_is_accepted(model):
    """The shape upstream's pipeline actually passes: ``ids.unsqueeze(0).expand(batch, -1)``."""
    torch.manual_seed(0)
    with torch.no_grad():
        out, _ = model(**_inputs(batch=2), current_temporal_ids=torch.tensor([[2], [2]]))
    assert out.shape == (2, 64, FRAMES, GRID_H, GRID_W)


def test_temporal_ids_must_cover_every_latent_frame(model):
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="must have 2 entries"):
        with torch.no_grad():
            model(
                hidden_states=_latent(frames=2),
                timestep=torch.full((1,), 1000.0),
                encoder_hidden_states=torch.randn(1, TEXT_LEN, TINY["text_states_dim"]),
                current_temporal_ids=torch.tensor([[0]]),
            )


def test_temporal_ids_shift_the_output(model):
    """Position ids are load-bearing, not decorative.

    The window renumbers positions every chunk, so a port that ignored ``current_temporal_ids`` and
    always used ``arange(T)`` would produce plausible video with the chunk placed at position 0
    forever.
    """
    torch.manual_seed(0)
    inputs = _inputs()
    with torch.no_grad():
        at_zero, _ = model(**inputs, current_temporal_ids=torch.tensor([[0]]))
        at_two, _ = model(**inputs, current_temporal_ids=torch.tensor([[2]]))
    assert not torch.allclose(at_zero, at_two, rtol=1e-3, atol=1e-3)


def test_causal_declares_the_rollout_and_changes_no_computation():
    """``causal=True`` is the shipped value and must not alter a single forward.

    The name is the trap. Upstream's DiT reads the flag only to require ``chunk_size``; attention is
    ``is_causal=False`` at both call sites, and the real consumers are outside the DiT (the pipeline
    gates its sliding-window VAE encode and its ``chunk_size`` on it, the streaming session refuses a
    config without it). So a port that built an intra-forward mask when the flag was set would be
    *more* faithful-looking and numerically wrong on every production request.
    """
    torch.manual_seed(0)
    declared = JoyAIVideoEditTransformer3DModel(**TINY, causal=True, chunk_size=1, dtype=torch.float32).eval()
    torch.manual_seed(0)
    plain = JoyAIVideoEditTransformer3DModel(**TINY, causal=False, dtype=torch.float32).eval()
    assert declared.causal is True

    torch.manual_seed(0)
    inputs = _inputs()
    with torch.no_grad():
        assert torch.equal(declared(**inputs)[0], plain(**inputs)[0])


def test_causal_without_a_chunk_size_is_refused():
    """Upstream's own validation (``dit.py:551``): a chunk-autoregressive rollout needs a chunk size.

    Defaulting to "one chunk covering everything" instead would silently turn the KV window off and
    produce a single non-autoregressive pass -- plausible output, no error.
    """
    with pytest.raises(ValueError, match="`chunk_size` must be provided"):
        JoyAIVideoEditTransformer3DModel(**TINY, causal=True, chunk_size=None)


def test_the_real_config_declares_the_causal_rollout():
    """``CAUSAL`` is ``True`` in ``config.py:33``, paired with ``chunk_size=1``.

    Pinned because the pipeline's sliding-window VAE encode is gated on it: flipping this to ``False``
    would skip that encode and feed the DiT latents computed over the whole clip at once.
    """
    config = dit_config()
    assert config["causal"] is True
    assert config["chunk_size"] == 1


def test_rope_dims_must_sum_to_the_head_dimension():
    with pytest.raises(ValueError, match="must equal head_dim"):
        JoyAIVideoEditTransformer3DModel(**{**TINY, "rope_dim_list": [16, 56, 32]})


def test_unpatchify_inverts_the_token_flattening(model):
    """``[B, T*H*W, C] -> [B, C, T, H, W]`` in frame-major order.

    ``img_in(x).flatten(2).transpose(1, 2)`` emits all tokens of frame 0, then frame 1; a
    ``reshape`` that assumed channel-major here would transpose the video without changing its shape.
    """
    tokens = torch.arange(2 * 3 * TOKENS_PER_FRAME * 64, dtype=torch.float32).reshape(2, 3 * TOKENS_PER_FRAME, 64)
    latent = model.unpatchify(tokens, 3, GRID_H, GRID_W)
    assert latent.shape == (2, 64, 3, GRID_H, GRID_W)
    # Token (frame f, row r, col c) must land at latent[:, :, f, r, c].
    for f, r, c in ((0, 0, 0), (1, 2, 4), (2, 1, 3)):
        torch.testing.assert_close(latent[:, :, f, r, c], tokens[:, f * TOKENS_PER_FRAME + r * GRID_W + c, :])


def test_unpatchify_rejects_a_token_count_that_does_not_fill_the_grid(model):
    with pytest.raises(ValueError, match="does not match grid"):
        model.unpatchify(torch.zeros(1, TOKENS_PER_FRAME + 1, 64), 1, GRID_H, GRID_W)


# --- KV window wiring ----------------------------------------------------------------------------


def _store_clean_chunk(model, window: JoyKVWindow, latent: torch.Tensor, chunk_id: int, text: torch.Tensor):
    """The third per-chunk forward: zero timestep, no reference, text stream skipped, write only."""
    with torch.no_grad():
        model(
            hidden_states=latent,
            timestep=torch.zeros(latent.shape[0]),
            encoder_hidden_states=text,
            current_temporal_ids=torch.tensor([[0]]),
            kv_window=window,
            kv_cache_mode="store",
            kv_cache_scope="cond",
            kv_cache_chunk_id=chunk_id,
            kv_cache_selected_chunk_ids=[],
            kv_cache_pre_rope=True,
            skip_text_stream=True,
        )


def test_only_the_image_stream_is_written_to_the_cache(model):
    """Cached entries are ``TOKENS_PER_FRAME`` long, not ``TOKENS_PER_FRAME + TEXT_LEN``.

    Text is fused by concatenation, so it is present in the attention but must never be cached --
    caching it would make the prompt accumulate once per chunk and silently reweight it.
    """
    torch.manual_seed(0)
    window = JoyKVWindow()
    text = torch.randn(1, TEXT_LEN, TINY["text_states_dim"])
    _store_clean_chunk(model, window, _latent(), chunk_id=0, text=text)

    assert window.resident_chunk_ids("cond") == {0}
    window.configure(scope="cond", mode="reuse", selected_chunk_ids=[0], pre_rope=True)
    for entry in window.read(0):
        assert entry["key"].shape[1] == TOKENS_PER_FRAME
        assert entry["pre_rope"] is True


def test_every_block_writes_its_own_cache_entry(model):
    """One entry per layer. A single shared entry would make all 40 layers attend to layer 39's keys."""
    torch.manual_seed(0)
    window = JoyKVWindow()
    _store_clean_chunk(model, window, _latent(), chunk_id=0, text=torch.randn(1, TEXT_LEN, TINY["text_states_dim"]))

    window.configure(scope="cond", mode="reuse", selected_chunk_ids=[0], pre_rope=True)
    keys = [window.read(layer)[0]["key"] for layer in range(TINY["mm_double_blocks_depth"])]
    assert all(k.shape == keys[0].shape for k in keys)
    assert not torch.allclose(keys[0], keys[1], rtol=1e-3, atol=1e-3)


def test_reusing_a_cached_chunk_changes_the_denoised_output(model):
    """The cache is actually consumed, not just populated.

    A port that stored keys and never prepended them would pass every KV-window unit test and produce
    temporally incoherent video with no error anywhere.
    """
    torch.manual_seed(0)
    text = torch.randn(1, TEXT_LEN, TINY["text_states_dim"])
    current = _latent()
    reference = _latent()

    window = JoyKVWindow()
    _store_clean_chunk(model, window, _latent(), chunk_id=0, text=text)

    denoise = {
        "hidden_states": current,
        "timestep": torch.full((1,), 1000.0),
        "encoder_hidden_states": text,
        "ref_video_latent": reference,
        "current_temporal_ids": torch.tensor([[1]]),
        "kv_cache_pre_rope": True,
    }
    with torch.no_grad():
        with_cache, _ = model(
            **denoise,
            cached_temporal_ids=torch.tensor([[0]]),
            kv_window=window,
            kv_cache_mode="reuse",
            kv_cache_scope="cond",
            kv_cache_chunk_id=0,
            kv_cache_selected_chunk_ids=[0],
        )
        without_cache, _ = model(**denoise)

    assert with_cache.shape == without_cache.shape
    assert not torch.allclose(with_cache, without_cache, rtol=1e-3, atol=1e-3)


def test_both_denoise_steps_of_a_chunk_see_the_same_assembled_cache(model):
    """The memoised assembly must be a pure cache, not a behaviour change.

    Both steps share one window and one position table, so the rotated cached keys are identical
    between them. Asserting the *output* is reproducible pins that the memo is keyed correctly -- a
    stale memo served across a position change is the failure this guards.
    """
    torch.manual_seed(0)
    text = torch.randn(1, TEXT_LEN, TINY["text_states_dim"])
    window = JoyKVWindow()
    _store_clean_chunk(model, window, _latent(), chunk_id=0, text=text)

    reuse = {
        "hidden_states": _latent(),
        "encoder_hidden_states": text,
        "current_temporal_ids": torch.tensor([[1]]),
        "cached_temporal_ids": torch.tensor([[0]]),
        "kv_window": window,
        "kv_cache_mode": "reuse",
        "kv_cache_scope": "cond",
        "kv_cache_chunk_id": 0,
        "kv_cache_selected_chunk_ids": [0],
        "kv_cache_pre_rope": True,
    }
    with torch.no_grad():
        first, _ = model(**reuse, timestep=torch.full((1,), 1000.0))
        again, _ = model(**reuse, timestep=torch.full((1,), 1000.0))
    assert torch.equal(first, again)

    # A different cached position must not be served from the memo.
    window_ids_moved = {
        **reuse,
        "cached_temporal_ids": torch.tensor([[1]]),
        "current_temporal_ids": torch.tensor([[2]]),
    }
    with torch.no_grad():
        moved, _ = model(**window_ids_moved, timestep=torch.full((1,), 1000.0))
    assert not torch.allclose(first, moved, rtol=1e-3, atol=1e-3)


def test_no_cache_is_written_on_a_reuse_only_forward(model):
    """``mode="reuse"`` must not mutate the store.

    The denoise steps run twice per chunk; writing on them would overwrite the clean-latent entry
    with a *noisy* one, which is silently wrong rather than an error.
    """
    torch.manual_seed(0)
    text = torch.randn(1, TEXT_LEN, TINY["text_states_dim"])
    window = JoyKVWindow()
    _store_clean_chunk(model, window, _latent(), chunk_id=0, text=text)
    generation = window.generation

    with torch.no_grad():
        model(
            hidden_states=_latent(),
            timestep=torch.full((1,), 1000.0),
            encoder_hidden_states=text,
            current_temporal_ids=torch.tensor([[1]]),
            cached_temporal_ids=torch.tensor([[0]]),
            kv_window=window,
            kv_cache_mode="reuse",
            kv_cache_scope="cond",
            kv_cache_chunk_id=0,
            kv_cache_selected_chunk_ids=[0],
            kv_cache_pre_rope=True,
        )
    assert window.generation == generation
    assert window.resident_chunk_ids("cond") == {0}
