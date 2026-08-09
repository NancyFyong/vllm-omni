# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from https://github.com/jd-opensource/JoyAI-Video-Edit
# (``deploy/xvideo/models/vae/vae.py``, class ``XVAEChunkCausal``). Upstream is
# Apache-2.0.
"""JoyAI-Video-Edit's chunk-causal video VAE.

Compression is **24x spatial** and **8x temporal**, which is not what the config file appears to say.
``block_in_channels`` has four entries and ``patch_size`` is 2, so upstream's own
``ffactor_spatial = patch_size * 2 ** (len(temporal_downsample) - 1)`` evaluates to 16 -- but that
factor describes only the part of the network *after* the ``Stem``. The full chain is::

    Stem (x2/3)  ->  patchify (/2)  ->  3 downsample blocks (/8)   =  /24 spatial
    (temporal untouched)              3 temporal downsamples (/8)  =  /8  temporal

and on the way back ``decoder (x8) -> unpatchify (x2) -> Head (x1.5) = x24``. So 720x1248 pixels
correspond to a 30x52 latent, and ``1 + 8n`` pixel frames to ``1 + n`` latent frames.

**The Stem bypass.** Upstream's ``Stem.forward`` silently ``return x``-es when the input is not
divisible by its stride of 3, and upstream's divisibility asserts run on the *post*-Stem shape, so
they cannot catch it. Worse, the bypass is dimensionally consistent: with the Stem the channel count
comes back out at 3 (``3*9 -> 3*4`` then ``pixel_shuffle(2)``), and without it the channel count was
never 3-anything-else -- so ``patchify`` yields the 12 channels the encoder wants either way. At
1280 pixels wide, ``1280 % 3 == 2`` bypasses the Stem while ``1280 % 16 == 0`` satisfies the assert,
and the whole model runs to completion on an 80-wide latent instead of a 52-wide one. That is why
upstream's server default width is 1248 and not 1280. Here the bypass raises instead
(:func:`validate_pixel_shape`, plus a total guard inside :class:`Stem`).

**Chunk-causal, not frame-causal.** The name is precise and the distinction is load-bearing.
``ChunkCausalConv3d`` prepends the previous chunk's last frame but pads the *back* of the window by
replicating its own last frame, so the final frame of a chunk sees a fabricated future. Causality
therefore holds only at chunk boundaries: encoding frames ``[0, 9)`` and encoding ``[0, 17)`` give
**bit-identical** latents for the overlapping range when the shorter clip ends on a chunk boundary,
and measurably different ones (max abs diff ~0.7 in the port's own check) when it does not. Anything
feeding this VAE incrementally must therefore hand it chunk-aligned groups of frames.

Upstream's separate ``vae_compile`` wrapper is ported, in ``joyai_video_edit_vae_compile.py``, but is
**off by default**: profiling put the whole of this port's throughput gap to upstream in the VAE, and
the compile is what closes it, yet ``max-autotune-no-cudagraphs`` over 66 ``Conv3d`` layers costs more
autotune time than a single offline request can earn back. That module owns the cost model and the
``VLLM_OMNI_JOYAI_VAE_COMPILE`` switch; nothing in this file changes when it is enabled, since it wraps
``_encode``/``_decode`` from outside.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import nn

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION,
)

# Number of trailing temporal frames each causal conv carries across a chunk boundary.
CACHE_T = 1

# ``Stem`` hyper-parameters. Upstream hardcodes these as defaults; they are named here because
# ``stride`` is half of why the true spatial factor is 24 rather than 16.
STEM_STRIDE = 3
STEM_GROUP = 2
# ``Head`` upsamples by this much on the way out, the other half of the 24x.
HEAD_SCALE = 1.5


def validate_pixel_shape(num_frames: int, height: int, width: int) -> None:
    """Reject pixel geometries the VAE cannot represent, in *pixel* space.

    ``height``/``width`` must be divisible by 24 -- not by the 16 the config's ``ffactor_spatial``
    suggests. 16 is the constraint on the post-``Stem`` shape; 24 is its pre-image through the
    ``Stem``'s 2/3 resize, and it is strictly stronger (720, 1248 pass; 1280 does not). Checking only
    the weaker one is how a 1280-wide request silently produces a wrong-resolution latent.

    ``num_frames`` must be ``1 + 8n``: the causal encoder handles the first frame on its own and then
    consumes whole groups of 8.
    """
    if num_frames % VAE_TEMPORAL_COMPRESSION != 1:
        raise ValueError(
            f"num_frames must be 1 + {VAE_TEMPORAL_COMPRESSION}n (got {num_frames}); "
            f"the nearest valid values are "
            f"{1 + VAE_TEMPORAL_COMPRESSION * (num_frames // VAE_TEMPORAL_COMPRESSION)} and "
            f"{1 + VAE_TEMPORAL_COMPRESSION * (num_frames // VAE_TEMPORAL_COMPRESSION + 1)}."
        )
    bad = [name for name, size in (("height", height), ("width", width)) if size % VAE_SPATIAL_COMPRESSION != 0]
    if bad:
        raise ValueError(
            f"height and width must be divisible by {VAE_SPATIAL_COMPRESSION} "
            f"(got height={height}, width={width}; offending: {', '.join(bad)}). "
            f"Note this is stricter than the VAE's internal post-Stem factor of "
            f"{VAE_SPATIAL_COMPRESSION * STEM_GROUP // STEM_STRIDE}: a size divisible by that but not "
            f"by {VAE_SPATIAL_COMPRESSION} bypasses the Stem and produces a wrong-resolution latent "
            f"without any error (e.g. width 1280, which is why the reference default is 1248)."
        )


def num_latent_frames(num_frames: int) -> int:
    """Latent frame count for a validated pixel frame count."""
    return 1 + (num_frames - 1) // VAE_TEMPORAL_COMPRESSION


def latent_spatial_shape(height: int, width: int) -> tuple[int, int]:
    """Latent ``(height, width)`` for a validated pixel resolution."""
    return height // VAE_SPATIAL_COMPRESSION, width // VAE_SPATIAL_COMPRESSION


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """Channel-wise RMS norm as upstream writes it: ``normalize(x) * sqrt(dim) * gamma + bias``.

    ``F.normalize`` divides by the L2 norm, so the ``sqrt(dim)`` factor is what turns it into an RMS
    normalisation. Not interchangeable with :class:`torch.nn.RMSNorm`, which normalises over the last
    dimension; here ``dim=1`` (channels) and ``gamma`` carries trailing singleton axes so it
    broadcasts over ``(t, h, w)``.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = False, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1 if self.channel_first else -1) * self.scale * self.gamma + self.bias


class AttnBlock(nn.Module):
    """Spatial self-attention applied independently per frame (frames fold into the batch)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMSNorm(in_channels, channel_first=True, images=False)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, t, h, w = x.shape

        x = self.norm(x)
        q = rearrange(self.q(x), "b c t h w -> (b t) 1 (h w) c")
        k = rearrange(self.k(x), "b c t h w -> (b t) 1 (h w) c")
        v = rearrange(self.v(x), "b c t h w -> (b t) 1 (h w) c")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "(b t) 1 (h w) c -> b c t h w", b=b, t=t, h=h, w=w)
        return self.proj_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(x)


class ChunkCausalConv3d(nn.Conv3d):
    """Conv3d whose temporal receptive field reaches backwards into the previous chunk.

    Temporal padding is supplied explicitly rather than by ``Conv3d``: the frame *before* the window
    comes from ``cache_x`` (the previous chunk's last frame) so that streaming chunk-by-chunk gives
    the same result as a single pass, and the frame *after* is a replication of the last input frame,
    which is what keeps the layer causal.
    """

    def __init__(
        self,
        chunk_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.chunk_size = chunk_size
        if self.padding[0] != 1:
            raise ValueError(f"Causal padding only supports temporal padding of 1, got {self.padding[0]}.")
        # Spatial padding stays with F.pad; temporal padding is done by concatenation below.
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if cache_x is not None:
            padding_front = cache_x.to(x.device)
        elif x.shape[2] == 1:
            # First chunk of a stream: the frame before frame 0 is frame 0 itself.
            padding_front = x
        else:
            raise ValueError(f"Temporal dimension must be 1 when cache_x is None, got {x.shape[2]}.")
        x = torch.cat([padding_front, x, x[:, :, -1:, :, :]], dim=2)
        return super().forward(F.pad(x, list(self._padding)))


class ResidualBlock(nn.Module):
    def __init__(self, chunk_size: int, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = RMSNorm(in_channels, channel_first=True, images=False)
        self.conv1 = ChunkCausalConv3d(chunk_size, in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = RMSNorm(out_channels, channel_first=True, images=False)
        self.conv2 = ChunkCausalConv3d(chunk_size, out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = ChunkCausalConv3d(
                chunk_size, in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        shortcut = x

        x = swish(self.norm1(x))
        x = _causal_conv(self.conv1, x, feat_cache, feat_idx)
        x = swish(self.norm2(x))
        x = _causal_conv(self.conv2, x, feat_cache, feat_idx)

        if self.in_channels != self.out_channels:
            shortcut = self.nin_shortcut(shortcut)
        return x + shortcut


def _causal_conv(
    conv: ChunkCausalConv3d,
    x: torch.Tensor,
    feat_cache: list[torch.Tensor | None] | None,
    feat_idx: list[int] | None,
) -> torch.Tensor:
    """Run ``conv``, threading the cross-chunk temporal cache through it.

    ``feat_idx`` is a one-element list used as a mutable counter: every causal conv in the encoder
    (or decoder) consumes exactly one cache slot, in call order, and the count must match the slot
    count allocated in :meth:`JoyAIVideoEditVAE.clear_cache`. The input is cached *before* the conv
    runs, so the slot holds this chunk's last input frame for the next chunk to prepend.
    """
    if feat_cache is None or feat_idx is None:
        return conv(x)
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    out = conv(x, cache_x=feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return out


class DownsampleBlock(nn.Module):
    """Space-to-depth downsample with a channel-averaged shortcut.

    The conv emits ``out_channels // factor`` channels and the pixel-unshuffle-style ``rearrange``
    folds the discarded resolution back into channels, so no information is thrown away by the main
    path. The shortcut path unshuffles the *input* and averages groups of channels down to the same
    width.
    """

    def __init__(self, chunk_size: int, in_channels: int, out_channels: int, temporal_downsample: bool):
        super().__init__()
        factor = 8 if temporal_downsample else 4
        self.conv = ChunkCausalConv3d(
            chunk_size, in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1
        )

        self.temporal_downsample = temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        r1 = 2 if self.temporal_downsample else 1

        # The first chunk carries a single frame, which is odd; duplicating frame 0 makes the
        # temporal fold well-defined and leaves the chunk at one latent frame.
        if self.temporal_downsample and first_chunk:
            shortcut = torch.cat([x[:, :, :1, :, :], x], dim=2)
        else:
            shortcut = x
        shortcut = rearrange(shortcut, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        x = _causal_conv(self.conv, x, feat_cache, feat_idx)

        if self.temporal_downsample and first_chunk:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)
        x = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)

        b, _c, t, h, w = shortcut.shape
        shortcut = shortcut.view(b, x.shape[1], self.group_size, t, h, w).mean(dim=2)
        return x + shortcut


class UpsampleBlock(nn.Module):
    """Depth-to-space upsample with a channel-repeated shortcut (mirror of :class:`DownsampleBlock`)."""

    def __init__(self, chunk_size: int, in_channels: int, out_channels: int, temporal_upsample: bool):
        super().__init__()
        factor = 8 if temporal_upsample else 4
        self.conv = ChunkCausalConv3d(
            chunk_size, in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1
        )

        self.temporal_upsample = temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        r1 = 2 if self.temporal_upsample else 1

        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)

        x = _causal_conv(self.conv, x, feat_cache, feat_idx)
        x = rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)

        x = x + shortcut
        if self.temporal_upsample and first_chunk:
            # Undoes the frame-0 duplication the encoder's first chunk introduced.
            x = x[:, :, 1:, :, :]
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
        chunk_size: int,
    ):
        super().__init__()

        self.z_channels = z_channels
        self.block_in_channels = block_in_channels
        self.num_res_blocks = num_res_blocks

        cur_chunk_size = chunk_size
        self.conv_in = ChunkCausalConv3d(
            cur_chunk_size, in_channels, block_in_channels[0], kernel_size=3, stride=1, padding=1
        )

        # Flat ModuleList (not nested per level) because the checkpoint is keyed that way:
        # ``encoder.down_blocks.<n>....``, with residual and downsample blocks interleaved.
        self.down_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.down_blocks.append(
                    DownsampleBlock(cur_chunk_size, block_in, block_out, temporal_downsample[i_level])
                )
                if temporal_downsample[i_level]:
                    cur_chunk_size //= 2

        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in),
                AttnBlock(block_in),
                ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in),
            ]
        )

        self.norm_out = RMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = ChunkCausalConv3d(cur_chunk_size, block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x = _causal_conv(self.conv_in, x, feat_cache, feat_idx)

        for block in self.down_blocks:
            if isinstance(block, DownsampleBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        x = swish(self.norm_out(x))
        return _causal_conv(self.conv_out, x, feat_cache, feat_idx)


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        num_res_blocks: int,
        block_in_channels: tuple[int, ...],
        temporal_upsample: tuple[bool, ...],
        chunk_size: int,
    ):
        super().__init__()

        self.z_channels = z_channels
        self.block_in_channels = block_in_channels
        self.num_res_blocks = num_res_blocks

        cur_chunk_size = chunk_size // (2 ** sum(temporal_upsample))
        block_in = block_in_channels[0]
        self.conv_in = ChunkCausalConv3d(cur_chunk_size, z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in),
                AttnBlock(block_in),
                ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in),
            ]
        )

        # One extra residual block per level than the encoder, matching upstream (and the checkpoint).
        self.up_blocks = nn.ModuleList([])
        for i_level, block_in in enumerate(block_in_channels):
            for _ in range(self.num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(cur_chunk_size, in_channels=block_in, out_channels=block_in))

            if i_level != len(block_in_channels) - 1:
                block_out = block_in_channels[i_level + 1]
                self.up_blocks.append(UpsampleBlock(cur_chunk_size, block_in, block_out, temporal_upsample[i_level]))
                if temporal_upsample[i_level]:
                    cur_chunk_size *= 2

        self.norm_out = RMSNorm(block_in, channel_first=True, images=False)
        self.conv_out = ChunkCausalConv3d(cur_chunk_size, block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x = _causal_conv(self.conv_in, x, feat_cache, feat_idx)

        for block in self.mid_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = block(x)

        for block in self.up_blocks:
            if isinstance(block, UpsampleBlock):
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
            else:
                x = block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        x = swish(self.norm_out(x))
        return _causal_conv(self.conv_out, x, feat_cache, feat_idx)


class Stem(nn.Module):
    """Learned 2/3 resize in front of the encoder -- the reason spatial compression is 24x, not 16x.

    ``pixel_unshuffle(3) -> 1x1 conv -> pixel_shuffle(2)`` maps ``c`` channels to ``c`` channels while
    scaling both spatial axes by ``group / stride == 2/3``.
    """

    def __init__(self, channels: int, stride: int = STEM_STRIDE, group: int = STEM_GROUP):
        super().__init__()
        self.stride = stride
        self.group = group
        self.proj = nn.Conv2d(channels * stride * stride, channels * group * group, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if h % self.stride != 0 or w % self.stride != 0:
            # Upstream returns ``x`` unchanged here. Because the bypass keeps the channel count at
            # ``c``, every downstream shape check still passes and the model produces a latent at the
            # wrong resolution -- so this raises instead. See the module docstring.
            raise ValueError(
                f"Stem requires spatial dims divisible by {self.stride}, got height={h}, width={w}. "
                f"Call validate_pixel_shape() before encoding."
            )
        oh, ow = h * self.group // self.stride, w * self.group // self.stride
        z = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        z = F.pixel_unshuffle(z, self.stride)
        z = self.proj(z)
        z = F.pixel_shuffle(z, self.group)
        return z.reshape(b, t, c, oh, ow).permute(0, 2, 1, 3, 4)


class HeadResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.act(self.dw(x)))


class Head(nn.Module):
    """Per-frame 1.5x refinement upsampler on the decoder output -- the mirror of the ``Stem``.

    A bilinear upsample of the decoder output carries the signal; the conv stack only predicts a
    residual on top of it. Output is clamped to the ``[-1, 1]`` pixel range.
    """

    def __init__(
        self,
        channels: int,
        scale: float = HEAD_SCALE,
        hidden: int = 32,
        num_blocks: int = 4,
        mid_channels: int = 12,
    ):
        super().__init__()
        self.scale = float(scale)
        self.conv_in = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.blocks = nn.Sequential(*[HeadResBlock(hidden) for _ in range(num_blocks)])
        self.reduce = nn.Conv2d(hidden, mid_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        oh, ow = round(h * self.scale), round(w * self.scale)
        z = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        f = self.act(self.conv_in(z))
        f = self.blocks(f)
        f = self.reduce(f)
        f = F.interpolate(f, size=(oh, ow), mode="bilinear", align_corners=False)
        residual = self.conv_out(f)
        base = F.interpolate(z, size=(oh, ow), mode="bilinear", align_corners=False)
        out = (base + residual).clamp(-1.0, 1.0)
        return out.reshape(b, t, c, oh, ow).permute(0, 2, 1, 3, 4)


class JoyAIVideoEditVAE(ModelMixin, ConfigMixin):
    """``XVAEChunkCausal`` -- JoyAI-Video-Edit's causal video VAE.

    ``encode``/``decode`` walk the clip chunk by chunk (first frame alone, then ``chunk_size`` frames
    at a time) with a per-conv temporal cache, so a chunked pass equals a single pass. Latent
    mean/std normalisation is *not* applied here: upstream's pipeline does it around these calls, and
    that is where this port keeps it too.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        latent_channels: int,
        layers_per_block: int,
        block_in_channels: tuple[int, ...],
        temporal_downsample: tuple[bool, ...],
        chunk_size: int,
        latents_mean: tuple[float, ...] | None = None,
        latents_std: tuple[float, ...] | None = None,
        enable_slicing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        # Post-Stem factors only; the Stem and Head supply the remaining 3/2 each way.
        self.ffactor_spatial = patch_size * 2 ** (len(temporal_downsample) - 1)
        self.ffactor_temporal = 2 ** sum(temporal_downsample)
        self.chunk_size = chunk_size
        # Kept as plain lists, deliberately not buffers: registering them would add keys the
        # checkpoint does not have and break a ``strict=True`` load.
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        expected_spatial = self.ffactor_spatial * STEM_STRIDE // STEM_GROUP
        if expected_spatial != VAE_SPATIAL_COMPRESSION or self.ffactor_temporal != VAE_TEMPORAL_COMPRESSION:
            raise ValueError(
                f"Config implies {expected_spatial}x spatial / {self.ffactor_temporal}x temporal "
                f"compression, but the pipeline is built around "
                f"{VAE_SPATIAL_COMPRESSION}x / {VAE_TEMPORAL_COMPRESSION}x. Every request-geometry "
                f"check would be wrong."
            )

        self.stem = Stem(in_channels)

        self.encoder = Encoder(
            in_channels=in_channels * (patch_size**2),
            z_channels=latent_channels,
            num_res_blocks=layers_per_block,
            block_in_channels=block_in_channels,
            temporal_downsample=temporal_downsample,
            chunk_size=chunk_size,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels * (patch_size**2),
            num_res_blocks=layers_per_block,
            block_in_channels=tuple(reversed(block_in_channels)),
            temporal_upsample=temporal_downsample,
            chunk_size=chunk_size,
        )

        self.head = Head(out_channels)

        self.use_slicing = enable_slicing

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    @staticmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Space-to-depth in front of the encoder.

        Channel order is ``(c r1 r2)`` here but ``(r1 r2 c)`` in :meth:`unpatchify`. That asymmetry is
        upstream's and must be preserved: encode and decode are separate learned paths, so the
        weights absorb it -- "fixing" it to be a true inverse would silently permute channels.
        """
        if patch_size == 1:
            return x
        return rearrange(x, "b c t (h r1) (w r2) -> b (c r1 r2) t h w", r1=patch_size, r2=patch_size)

    @staticmethod
    def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Depth-to-space behind the decoder. See :meth:`patchify` on the channel ordering."""
        if patch_size == 1:
            return x
        return rearrange(x, "b (r1 r2 c) t h w -> b c t (h r1) (w r2)", r1=patch_size, r2=patch_size)

    def clear_cache(self) -> None:
        """Allocate one temporal-cache slot per causal conv and rewind the counters."""
        if not hasattr(self, "_enc_conv_num") or not hasattr(self, "_dec_conv_num"):
            self._enc_conv_num = sum(isinstance(m, ChunkCausalConv3d) for m in self.encoder.modules())
            self._dec_conv_num = sum(isinstance(m, ChunkCausalConv3d) for m in self.decoder.modules())
        self._enc_conv_idx = [0]
        self._dec_conv_idx = [0]
        self._enc_feat_map: list[torch.Tensor | None] = [None] * self._enc_conv_num
        self._dec_feat_map: list[torch.Tensor | None] = [None] * self._dec_conv_num

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        validate_pixel_shape(x.shape[2], x.shape[3], x.shape[4])
        x = self.stem(x)
        if x.shape[2] % self.ffactor_temporal != 1:
            raise ValueError(f"Post-Stem temporal dim must be {self.ffactor_temporal}n + 1, got {tuple(x.shape)}.")
        if x.shape[3] % self.ffactor_spatial != 0 or x.shape[4] % self.ffactor_spatial != 0:
            raise ValueError(
                f"Post-Stem spatial dims must be divisible by {self.ffactor_spatial}, got {tuple(x.shape)}."
            )
        x = self.patchify(x, self.patch_size)

        out = []
        self.clear_cache()
        iter_ = 1 + math.ceil((x.shape[2] - 1) / self.chunk_size)
        for i in range(iter_):
            self._enc_conv_idx = [0]
            frames = x[:, :, :1, :, :] if i == 0 else x[:, :, 1 + (i - 1) * self.chunk_size : 1 + i * self.chunk_size]
            out.append(
                self.encoder(
                    frames,
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    first_chunk=(i == 0),
                )
            )
        result = torch.cat(out, dim=2)
        self.clear_cache()
        return result

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if x.ndim != 5:
            raise ValueError(f"Input tensor must be 5D (b, c, t, h, w), got {tuple(x.shape)}.")

        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        latent_chunk_size = self.chunk_size // self.ffactor_temporal

        self.clear_cache()
        decoded = []
        iter_ = 1 + math.ceil((z.shape[2] - 1) / latent_chunk_size)
        for i in range(iter_):
            self._dec_conv_idx = [0]
            frames = (
                z[:, :, :1, :, :] if i == 0 else z[:, :, 1 + (i - 1) * latent_chunk_size : 1 + i * latent_chunk_size]
            )
            decoded.append(
                self.decoder(
                    frames,
                    feat_cache=self._dec_feat_map,
                    feat_idx=self._dec_conv_idx,
                    first_chunk=(i == 0),
                )
            )
        result = torch.cat(decoded, dim=2)
        self.clear_cache()

        return self.head(self.unpatchify(result, self.patch_size))

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
    ):
        """Round-trip a clip. Used by tests; the pipeline calls ``encode``/``decode`` separately."""
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec) if return_dict else (dec, posterior)


# Upstream's class name, kept as an alias so ``vae/config.json``'s ``_class_name`` resolves.
XVAEChunkCausal = JoyAIVideoEditVAE
