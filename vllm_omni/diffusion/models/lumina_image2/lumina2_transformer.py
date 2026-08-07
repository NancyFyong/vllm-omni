# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py

from collections.abc import Iterable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import (
    LuminaLayerNormContinuous,
    LuminaRMSNormZero,
    RMSNorm,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def _get_sequence_parallel_world_size_or_one() -> int:
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

        return max(1, int(get_sequence_parallel_world_size()))
    except Exception:
        return 1


def _build_joint_padding_mask(
    reference: torch.Tensor,
    seq_lengths: list[int],
    padded_len: int,
) -> torch.Tensor:
    """Full-length key-padding mask for the joint ``[caption; image]`` sequence.

    ``True`` marks valid tokens; padded / cross-sample positions are ``False``.
    Built as a plain tensor (not a module output) so Ulysses sequence parallelism
    never shards it — the all-to-all reconstructs the full sequence per head shard,
    so the mask must match the full post-all-to-all query length. ``reference``
    only supplies device/batch; ``padded_len`` is the SP-padded joint length.
    """
    batch_size = reference.shape[0]
    attention_mask = reference.new_zeros(batch_size, padded_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lengths):
        attention_mask[i, :seq_len] = True
    return attention_mask


def _is_lumina_transformer_block(name: str, module: object) -> bool:
    """HSDP shard condition: numbered blocks under the main ``layers`` stack.

    Lumina names its deep block stack ``layers`` (not ``transformer_blocks``),
    so the shared ``is_transformer_block_module`` helper does not match; this
    targets ``layers.<idx>`` while leaving the small refiner stacks replicated.
    """
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit()


if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        cap_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, hidden_size, bias=True)
        )

    def forward(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).type_as(hidden_states)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(encoder_hidden_states)
        return time_embed, caption_embed


class Lumina2FeedForward(nn.Module):
    """Gated SwiGLU MLP (diffusers ``LuminaFeedForward``) with tensor-parallel linears.

    ``linear_1`` (gate) and ``linear_3`` (up) are column-parallel and shard the
    intermediate dimension identically, so their element-wise product stays
    consistent per shard; ``linear_2`` (down) is row-parallel and reduces across
    the sharded intermediate dimension. Parameter names match the diffusers
    checkpoint (``linear_1``/``linear_2``/``linear_3``) so weights load unchanged,
    and the intermediate-dim computation reproduces ``LuminaFeedForward`` exactly.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ) -> None:
        super().__init__()
        # diffusers ``LuminaFeedForward`` does NOT apply the classic LLaMA 2/3
        # SwiGLU reduction: it takes ``inner_dim`` (the block passes ``4 * dim``)
        # as-is, optionally scales by ``ffn_dim_multiplier``, then rounds up to
        # ``multiple_of``. Reproducing that exactly is required or the checkpoint
        # FFN weights (shape ``4*dim``) get silently truncated.
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = ColumnParallelLinear(dim, inner_dim, bias=False, return_bias=False)
        self.linear_2 = RowParallelLinear(inner_dim, dim, bias=False, return_bias=False)
        self.linear_3 = ColumnParallelLinear(dim, inner_dim, bias=False, return_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # diffusers uses FP32SiLU (gate activation computed in fp32) — match it
        # for bit-level parity rather than the bf16 ``F.silu``.
        gate = self.linear_1(hidden_states)
        gate = F.silu(gate.float()).to(gate.dtype)
        return self.linear_2(gate * self.linear_3(hidden_states))


class Lumina2Attention(nn.Module):
    """Joint self-attention over the concatenated ``[caption; image]`` sequence.

    This replaces the diffusers ``Attention`` + ``Lumina2AttnProcessor2_0`` pair
    with vLLM-Omni's :class:`Attention` layer for the scaled-dot-product core,
    keeping RMS query/key norm, complex RoPE and GQA (via ``num_kv_heads``) as in
    the reference implementation. The three separate projections are fused into a
    tensor-parallel :class:`QKVParallelLinear` (checkpoint ``to_q``/``to_k``/``to_v``
    are merged via ``stacked_params_mapping`` in ``load_weights``) and the output
    projection is a :class:`RowParallelLinear` (checkpoint ``to_out.0``). Local
    (post-shard) head counts are read back from the fused layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        norm_eps: float = 1e-5,
        skip_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        inner_dim = num_attention_heads * self.head_dim

        # Fused, tensor-parallel QKV. GQA is expressed via total_num_kv_heads;
        # the weight_loader replicates KV heads across ranks when tp > num_kv_heads.
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_attention_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )
        # Local head counts after tensor-parallel sharding.
        self.heads = self.to_qkv.num_heads
        self.kv_heads = self.to_qkv.num_kv_heads

        # qk_norm="rms_norm" over the per-head dimension (elementwise affine).
        # head_dim is tensor-parallel invariant, so these stay replicated.
        self.norm_q = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)
        self.norm_k = RMSNorm(self.head_dim, eps=norm_eps, elementwise_affine=True)

        self.to_out = RowParallelLinear(inner_dim, dim, bias=False, return_bias=False)

        self.attn = Attention(
            num_heads=self.heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            num_kv_heads=self.kv_heads,
            skip_sequence_parallel=skip_sequence_parallel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.kv_heads, self.head_dim)

        dtype = query.dtype

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query = query.to(dtype)
        key = key.to(dtype)

        attn_metadata = None
        if attention_mask is not None:
            # vLLM-Omni flash-attn backend expects a 2D key-padding mask
            # (batch_size, seq_len) where True marks valid tokens.
            attn_metadata = AttentionMetadata(attn_mask=attention_mask.bool())

        # vLLM-Omni Attention handles GQA (num_kv_heads) internally.
        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).to(dtype)

        hidden_states = self.to_out(hidden_states)
        return hidden_states


class Lumina2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
        skip_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        self.attn = Lumina2Attention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            norm_eps=1e-5,
            skip_sequence_parallel=skip_sequence_parallel,
        )

        self.feed_forward = Lumina2FeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(
                embedding_dim=dim,
                norm_eps=norm_eps,
                norm_elementwise_affine=True,
            )
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(norm_hidden_states, attention_mask, image_rotary_emb)
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Lumina2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], axes_lens: list[int] = (300, 512, 512), patch_size: int = 2):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

        self.freqs_cis = self._precompute_freqs_cis(axes_dim, axes_lens, theta)

    def _precompute_freqs_cis(self, axes_dim: list[int], axes_lens: list[int], theta: int) -> list[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=self.theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return torch.cat(result, dim=-1).to(device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        image_seq_len = post_patch_height * post_patch_width
        device = hidden_states.device

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()
        seq_lengths = [cap_seq_len + image_seq_len for cap_seq_len in l_effective_cap_len]
        max_seq_len = max(seq_lengths)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add caption position ids
            position_ids[i, :cap_seq_len, 0] = torch.arange(cap_seq_len, dtype=torch.int32, device=device)
            position_ids[i, cap_seq_len:seq_len, 0] = cap_seq_len

            # add image position ids
            row_ids = (
                torch.arange(post_patch_height, dtype=torch.int32, device=device)
                .view(-1, 1)
                .repeat(1, post_patch_width)
                .flatten()
            )
            col_ids = (
                torch.arange(post_patch_width, dtype=torch.int32, device=device)
                .view(1, -1)
                .repeat(post_patch_height, 1)
                .flatten()
            )
            position_ids[i, cap_seq_len:seq_len, 1] = row_ids
            position_ids[i, cap_seq_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(
            batch_size, image_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            img_freqs_cis[i, :image_seq_len] = freqs_cis[i, cap_seq_len:seq_len]

        # image patch embeddings
        hidden_states = (
            hidden_states.view(batch_size, channels, post_patch_height, p, post_patch_width, p)
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )

        return hidden_states, cap_freqs_cis, img_freqs_cis, freqs_cis, l_effective_cap_len, seq_lengths


class Lumina2JointPrepare(nn.Module):
    """Assemble the joint ``[caption; image]`` sequence for the main block stack.

    This encapsulates the joint-sequence assembly (previously inline in the model
    ``forward``) into a module boundary so ``_sp_plan`` can shard its outputs for
    Ulysses sequence parallelism via ``split_output=True``. Under sequence
    parallelism the joint hidden states and per-position RoPE are padded to a
    length divisible by the SP world size (mirroring ``ernie_image``'s
    ``UnifiedPrepare``), then sharded along the sequence dim.

    The padding mask is intentionally **not** produced here: the Ulysses
    all-to-all inside the attention layer reconstructs the full sequence for each
    head shard, so the key-padding mask must stay full-length to match the
    post-all-to-all query. If it were a module output it would be sharded along
    with the hidden states, leaving a half-length mask. It is therefore built as
    a plain local tensor in :meth:`Lumina2Transformer2DModel.forward` (see
    :func:`_build_joint_padding_mask`), outside any SP-sharded module boundary.

    The module holds no parameters, so it is transparent to weight loading.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        max_seq_len = max(seq_lengths)

        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            joint_hidden_states[i, :encoder_seq_len] = encoder_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = hidden_states[i]

        sp_size = _get_sequence_parallel_world_size_or_one()
        pad_size = (-max_seq_len) % sp_size
        if pad_size:
            # Pad the sequence dim (dim 1) so it is divisible by the SP world size.
            # The RoPE tensor is complex (freqs_cis); zero padding is a no-op for
            # the padded (masked-out) positions. The matching full-length mask
            # built in ``forward`` keeps the pad tokens out of attention.
            joint_hidden_states = F.pad(joint_hidden_states, (0, 0, 0, pad_size))
            rotary_emb = F.pad(rotary_emb, (0, 0, 0, pad_size))

        return joint_hidden_states, rotary_emb


class Lumina2Transformer2DModel(nn.Module):
    r"""Lumina2NextDiT: a Transformer backbone for flow-matching text-to-image.

    Ported from diffusers ``Lumina2Transformer2DModel``. The diffusers mixins
    (``ModelMixin``/``ConfigMixin``/``PeftAdapterMixin``/``FromOriginalModelMixin``)
    are dropped; configuration is stored on ``self.config`` (a ``SimpleNamespace``)
    so the pipeline can read ``self.transformer.config.in_channels`` etc. The
    per-block attention uses vLLM-Omni's :class:`Attention` layer.
    """

    _no_split_modules = ["Lumina2TransformerBlock"]
    _repeated_blocks = ["Lumina2TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]
    # HSDP: shard the deep ``layers`` stack (small refiner stacks stay replicated).
    _hsdp_shard_conditions = [_is_lumina_transformer_block]
    # LoRA / fused-projection metadata for the tensor-parallel QKV.
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}
    # Cache-DiT over ``layers``: single-stream Pattern_3 (positional
    # attention_mask/rotary/temb, so introspection is off); separate cond/uncond
    # passes (has_separate_cfg) keep per-branch residual caches.
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"layers": ForwardPattern.Pattern_3},
        has_separate_cfg=True,
        check_forward_pattern=False,
    )
    # Ulysses SP: shard the two ``unified_prepare`` outputs (joint hidden states
    # and per-position RoPE) along the seq dim; ``norm_out`` gathers before
    # unpatchify. The padding mask is NOT a module output — it is built
    # full-length in ``forward`` so it stays aligned with the query after the
    # all-to-all. Mask + Ring is unsupported, so Ulysses-only (``ring_degree=1``).
    _sp_plan = {
        "unified_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
        "norm_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-5,
        scaling_factor: float = 1.0,
        axes_dim_rope: tuple[int, int, int] = (32, 32, 32),
        axes_lens: tuple[int, int, int] = (300, 512, 512),
        cap_feat_dim: int = 1024,
        od_config: OmniDiffusionConfig | None = None,
        quant_config: "QuantizationConfig | None" = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size

        # Config namespace consumed by the pipeline (in_channels, sample_size, ...).
        self.config = SimpleNamespace(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            scaling_factor=scaling_factor,
            axes_dim_rope=axes_dim_rope,
            axes_lens=axes_lens,
            cap_feat_dim=cap_feat_dim,
        )

        # 1. Positional, patch & conditional embeddings
        self.rope_embedder = Lumina2RotaryPosEmbed(
            theta=10000, axes_dim=axes_dim_rope, axes_lens=axes_lens, patch_size=patch_size
        )

        self.x_embedder = nn.Linear(in_features=patch_size * patch_size * in_channels, out_features=hidden_size)

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size, cap_feat_dim=cap_feat_dim, norm_eps=norm_eps
        )

        # 2. Noise and context refinement blocks
        # These run on the full (un-sharded) sequences *before* ``unified_prepare``
        # assembles the joint stream, i.e. outside the ``_sp_plan`` region, so under
        # Ulysses SP they must skip the all-to-all (``skip_sequence_parallel=True``).
        # Only the main joint ``layers`` operate on the SP-sharded sequence.
        self.noise_refiner = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                    skip_sequence_parallel=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                    skip_sequence_parallel=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # 3. Transformer blocks
        self.layers = nn.ModuleList(
            [
                Lumina2TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_layers)
            ]
        )

        # 3b. Joint-sequence assembly as a module boundary so ``_sp_plan`` can
        # shard its outputs for Ulysses sequence parallelism (holds no params).
        self.unified_prepare = Lumina2JointPrepare()

        # 4. Output norm & projection
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        # 1. Condition, positional & patch embedding
        batch_size, _, height, width = hidden_states.shape

        temb, encoder_hidden_states = self.time_caption_embed(hidden_states, timestep, encoder_hidden_states)

        (
            hidden_states,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(hidden_states, encoder_attention_mask)

        hidden_states = self.x_embedder(hidden_states)

        # 2. Context & noise refinement
        for layer in self.context_refiner:
            encoder_hidden_states = layer(encoder_hidden_states, encoder_attention_mask, context_rotary_emb)

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, None, noise_rotary_emb, temb)

        # 3. Joint Transformer blocks
        # Under SP the joint sequence is padded to a multiple of the SP world size
        # and a full-length padding mask applied. A single fixed-length prompt that
        # already divides the world size reduces to the mask-free single-stream path
        # (pad_size == 0, use_mask False), leaving the base path numerically unchanged.
        sp_size = _get_sequence_parallel_world_size_or_one()
        max_seq_len = max(seq_lengths)
        pad_size = (-max_seq_len) % sp_size
        padded_len = max_seq_len + pad_size
        # Mask only for genuine padding (SP pad_size > 0) or a ragged batch (unequal
        # seq_lengths); otherwise take the fast no-mask flash path even under SP. The
        # mask is a plain local tensor (NOT routed through the SP-sharded
        # ``unified_prepare``) so it stays full-length to match the post-all-to-all query.
        use_mask = pad_size > 0 or len(set(seq_lengths)) > 1
        attention_mask = _build_joint_padding_mask(hidden_states, seq_lengths, padded_len) if use_mask else None

        # ``unified_prepare`` assembles + pads the joint sequence; ``_sp_plan``
        # shards its two outputs (hidden_states, rotary_emb) along the seq dim.
        hidden_states, rotary_emb = self.unified_prepare(
            hidden_states,
            encoder_hidden_states,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)

        # 5. Unpatchify
        p = self.config.patch_size
        output = []
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            output.append(
                hidden_states[i][encoder_seq_len:seq_len]
                .view(height // p, width // p, p, p, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        output = torch.stack(output, dim=0)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Fuse the separate diffusers projections (``to_q``/``to_k``/``to_v``)
        # into the tensor-parallel ``to_qkv``; ``to_out.0`` maps to ``to_out``.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.replace("transformer.", "", 1)
            if ".to_out.0" in name:
                name = name.replace(".to_out.0", ".to_out")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params_dict.get(mapped)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                param = params_dict.get(name)
                if param is None:
                    logger.warning("Lumina2: unexpected transformer weight %s (skipped)", name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params
