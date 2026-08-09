# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JoyAI-Video-Edit's chunk-causal VAE.

Everything here runs on a tiny randomly-initialised config, because the properties under test are
structural: geometry, the shape guards, and the cross-chunk temporal cache. Each guards a failure
that produces finite, correctly-shaped output:

- The **24x** spatial factor. The config's own ``ffactor_spatial`` says 16 (it describes the network
  after the ``Stem``), so a port that trusts it computes latent shapes that are wrong by 3/2 while
  every tensor op still succeeds.
- The **Stem bypass**. Upstream silently skips the ``Stem`` when the input is not divisible by 3, and
  its divisibility asserts run *post*-Stem so they cannot catch it. Channel counts match either way,
  so the model runs to completion at the wrong resolution -- this is why width 1280 is invalid and
  1248 is the reference default.
- The **temporal cache slot accounting**. A miscount either raises (loudly, fine) or leaves a slot
  unused, which only shows up as a discontinuity at chunk boundaries several chunks in.
- **Chunk-causality**, asserted in both directions: bit-exact on chunk-aligned prefixes, and
  *deliberately not* exact otherwise. The second half stops a future "make it properly causal" change
  from silently diverging from the checkpoint's training-time behaviour.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit import joyai_video_edit_vae as vae_mod
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Real structure (4 levels, 3 temporal downsamples, patch_size 2) at toy widths, so the geometry and
# cache behaviour are the checkpoint's while the tests stay CPU-fast.
TINY_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "patch_size": 2,
    "latent_channels": 4,
    "layers_per_block": 1,
    "block_in_channels": [8, 16, 32, 64],
    "temporal_downsample": [True, True, True, False],
    "chunk_size": 8,
}
# Divisible by 24, deliberately non-square: a square clip hides transposed spatial axes.
TINY_HEIGHT, TINY_WIDTH = 24, 48


@pytest.fixture
def tiny_vae() -> vae_mod.JoyAIVideoEditVAE:
    torch.manual_seed(0)
    return vae_mod.JoyAIVideoEditVAE(**TINY_CONFIG).eval()


def _smooth_clip(num_frames: int) -> torch.Tensor:
    """Video-like content. Random noise would make the exactness assertions less meaningful."""
    t = torch.linspace(0, 1, num_frames)[:, None, None]
    h = torch.linspace(0, 1, TINY_HEIGHT)[None, :, None]
    w = torch.linspace(0, 1, TINY_WIDTH)[None, None, :]
    return torch.sin(6 * (h + w) + 3 * t).expand(3, num_frames, TINY_HEIGHT, TINY_WIDTH)[None].contiguous()


# --- geometry -----------------------------------------------------------------------------------


def test_spatial_factor_is_24_not_the_configs_16(tiny_vae):
    """``ffactor_spatial`` is the post-Stem factor; the Stem supplies the missing 3/2.

    Reading the config's number as the whole story is the single most likely porting mistake here,
    and it is invisible in every tensor shape downstream of the encoder.
    """
    assert tiny_vae.ffactor_spatial * vae_mod.STEM_STRIDE // vae_mod.STEM_GROUP == VAE_SPATIAL_COMPRESSION
    assert tiny_vae.ffactor_spatial != VAE_SPATIAL_COMPRESSION


def test_reference_resolution_maps_to_the_expected_latent_grid():
    """720x1248 -> 30x52. These two numbers appear in the DiT's token count, so pin them."""
    assert vae_mod.latent_spatial_shape(DEFAULT_HEIGHT, DEFAULT_WIDTH) == (30, 52)


def test_latent_frame_count_matches_the_1_plus_8n_layout():
    assert vae_mod.num_latent_frames(1) == 1
    assert vae_mod.num_latent_frames(9) == 2
    assert vae_mod.num_latent_frames(73) == 10


def test_encode_produces_the_predicted_latent_shape(tiny_vae):
    """The helpers used for request validation must agree with what the module actually emits."""
    clip = _smooth_clip(9)
    with torch.no_grad():
        latent = tiny_vae.encode(clip).latent_dist.mode()
    expected_h, expected_w = vae_mod.latent_spatial_shape(TINY_HEIGHT, TINY_WIDTH)
    assert tuple(latent.shape) == (
        1,
        TINY_CONFIG["latent_channels"],
        vae_mod.num_latent_frames(9),
        expected_h,
        expected_w,
    )


def test_decode_restores_the_original_pixel_shape(tiny_vae):
    """decoder x8 -> unpatchify x2 -> Head x1.5 must land back on 24x exactly, not off by rounding."""
    clip = _smooth_clip(9)
    with torch.no_grad():
        reconstructed = tiny_vae.decode(tiny_vae.encode(clip).latent_dist.mode()).sample
    assert tuple(reconstructed.shape) == tuple(clip.shape)


def test_decoder_output_is_clamped_to_the_pixel_range(tiny_vae):
    """``Head`` clamps to [-1, 1]; an unclamped port yields out-of-range pixels that wrap on cast."""
    with torch.no_grad():
        reconstructed = tiny_vae.decode(tiny_vae.encode(_smooth_clip(9)).latent_dist.mode()).sample
    assert reconstructed.min() >= -1.0 and reconstructed.max() <= 1.0


# --- shape guards -------------------------------------------------------------------------------


def test_width_1280_is_rejected_even_though_it_divides_the_post_stem_factor():
    """The load-bearing guard.

    1280 % 16 == 0 satisfies the VAE's *internal* assert while 1280 % 3 == 2 bypasses the Stem, so
    upstream accepts this request and returns an 80-wide latent instead of a 52-wide one with no
    error anywhere. Height 720 is valid, isolating width as the cause.
    """
    assert 1280 % (VAE_SPATIAL_COMPRESSION * vae_mod.STEM_GROUP // vae_mod.STEM_STRIDE) == 0
    with pytest.raises(ValueError, match="divisible by 24"):
        vae_mod.validate_pixel_shape(num_frames=9, height=DEFAULT_HEIGHT, width=1280)


def test_reference_default_resolution_is_accepted():
    """Counterpart to the 1280 case: the guard must not be so strict it rejects the real default."""
    vae_mod.validate_pixel_shape(num_frames=73, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH)


@pytest.mark.parametrize("num_frames", [2, 8, 16, 72])
def test_frame_counts_outside_1_plus_8n_are_rejected(num_frames):
    assert num_frames % VAE_TEMPORAL_COMPRESSION != 1
    with pytest.raises(ValueError, match="1 \\+ 8n"):
        vae_mod.validate_pixel_shape(num_frames=num_frames, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH)


def test_stem_raises_instead_of_bypassing_itself(tiny_vae):
    """Total guard at the point of failure, so ``_encode`` is not the only thing protecting it.

    A bypassed Stem is dimensionally consistent -- it emits ``in_channels`` either way, so
    ``patchify`` hands the encoder the channel count it expects regardless -- which is exactly why
    upstream's ``return x`` is silent.
    """
    with pytest.raises(ValueError, match="divisible by 3"):
        tiny_vae.stem(torch.zeros(1, 3, 1, TINY_HEIGHT, 80))


def test_encode_rejects_non_5d_input(tiny_vae):
    with pytest.raises(ValueError, match="must be 5D"):
        tiny_vae.encode(torch.zeros(3, 9, TINY_HEIGHT, TINY_WIDTH))


def test_config_implying_the_wrong_compression_is_rejected():
    """A config with a different level count would silently invalidate every geometry helper."""
    with pytest.raises(ValueError, match="spatial"):
        vae_mod.JoyAIVideoEditVAE(**{**TINY_CONFIG, "temporal_downsample": [True, True, False]})


# --- cross-chunk temporal cache -----------------------------------------------------------------


def test_every_allocated_cache_slot_is_consumed_exactly_once(tiny_vae):
    """Slot count must equal the number of causal convs actually invoked, per chunk.

    Too few slots raises an IndexError, but too many is silent: an unclaimed slot means some conv is
    not carrying its previous-chunk frame, which shows up only as a seam artefact at chunk
    boundaries.
    """
    clip = _smooth_clip(9)
    with torch.no_grad():
        patched = tiny_vae.patchify(tiny_vae.stem(clip), tiny_vae.patch_size)
        tiny_vae.clear_cache()
        tiny_vae.encoder(
            patched[:, :, :1],
            feat_cache=tiny_vae._enc_feat_map,
            feat_idx=tiny_vae._enc_conv_idx,
            first_chunk=True,
        )
    assert tiny_vae._enc_conv_idx[0] == tiny_vae._enc_conv_num
    assert all(slot is not None for slot in tiny_vae._enc_feat_map)


def test_chunk_aligned_prefix_encodes_bit_identically(tiny_vae):
    """Streaming equivalence, exactly.

    17 frames split as ``[0:1] [1:9] [9:17]`` and 9 frames split as ``[0:1] [1:9]``, so the shorter
    clip's chunk decomposition is a prefix of the longer one's. Every cross-chunk cache slot, its
    ordering, and the ``first_chunk`` frame-duplication must all be right for this to be exact --
    which is why it is asserted with ``torch.equal`` and not a tolerance.
    """
    clip = _smooth_clip(17)
    with torch.no_grad():
        short = tiny_vae.encode(clip[:, :, :9]).latent_dist.mode()
        long = tiny_vae.encode(clip).latent_dist.mode()
    assert torch.equal(short, long[:, :, : short.shape[2]])


def test_chunk_aligned_prefix_decodes_bit_identically(tiny_vae):
    """Same property on the way out, where ``UpsampleBlock``'s first-chunk frame trim is at stake."""
    clip = _smooth_clip(17)
    with torch.no_grad():
        latent = tiny_vae.encode(clip).latent_dist.mode()
        prefix = tiny_vae.decode(latent[:, :, :2]).sample
        full = tiny_vae.decode(latent).sample
    assert torch.equal(prefix, full[:, :, : prefix.shape[2]])


def test_unaligned_prefix_is_expected_to_differ(tiny_vae):
    """This VAE is chunk-causal, *not* frame-causal, and that is the trained behaviour.

    Convs replicate their own last frame as back-padding, so a chunk's final frame sees a fabricated
    future. Asserting the difference (rather than ignoring it) documents the contract for anything
    feeding the VAE incrementally, and fails if someone "fixes" the padding to be truly causal --
    which would diverge from the checkpoint while looking like an improvement.
    """
    big_chunk = vae_mod.JoyAIVideoEditVAE(**{**TINY_CONFIG, "chunk_size": 48}).eval()
    big_chunk.load_state_dict(tiny_vae.state_dict(), strict=True)
    clip = _smooth_clip(17)
    with torch.no_grad():
        short = big_chunk.encode(clip[:, :, :9]).latent_dist.mode()
        long = big_chunk.encode(clip).latent_dist.mode()
    # 9 frames -> one 8-frame chunk; 17 frames -> one 16-frame chunk. Not a prefix decomposition.
    assert not torch.equal(short, long[:, :, : short.shape[2]])


# --- patchify / unpatchify ----------------------------------------------------------------------


def test_patchify_and_unpatchify_use_different_channel_orders():
    """Upstream's asymmetry (``c r1 r2`` vs ``r1 r2 c``), preserved on purpose.

    Encode and decode are separate learned paths, so the weights absorb it. Making them true inverses
    permutes the decoder's input channels -- no shape change, garbage output.
    """
    x = torch.arange(2 * 3 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 3, 1, 4, 4)
    packed = vae_mod.JoyAIVideoEditVAE.patchify(x, 2)
    assert packed.shape == (2, 12, 1, 2, 2)
    assert not torch.equal(vae_mod.JoyAIVideoEditVAE.unpatchify(packed, 2), x)


def test_patchify_is_a_no_op_at_patch_size_one():
    x = torch.randn(1, 3, 1, 4, 4)
    assert torch.equal(vae_mod.JoyAIVideoEditVAE.patchify(x, 1), x)
    assert torch.equal(vae_mod.JoyAIVideoEditVAE.unpatchify(x, 1), x)


# --- normalisation ------------------------------------------------------------------------------


def test_latent_statistics_are_not_registered_as_buffers(tiny_vae):
    """They must stay plain attributes: buffers would add keys the checkpoint lacks.

    The DiT checkpoint is loaded ``strict=True`` deliberately, so an extra buffer here turns into a
    load failure -- and normalising latents is the pipeline's job, not the VAE's.
    """
    assert "latents_mean" not in dict(tiny_vae.named_buffers())
    assert "latents_std" not in dict(tiny_vae.named_buffers())
