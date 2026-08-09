# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""JoyAI-Video-Edit's dual-stream MMDiT (40 blocks, hidden 4096, 32 heads, 16.26B params).

Structurally a Wan-style AdaLN-single MMDiT, with three things that are specific to this model and
easy to get wrong:

**Text is fused by concatenation, not cross-attention.** Each block runs a full image stream and a
full text stream through *one* self-attention over ``cat([img, txt], dim=1)``. There is no
cross-attention module and no attention mask -- ``is_causal=False`` at every call site. Only the image
half of the concatenation is ever written to the KV cache.

**Reference (source) tokens share the target's positions.** ``ref_video_latent`` is appended to the
image stream with the *same* temporal ids as the noisy chunk, so source and target tokens occupy
identical ``(t, h, w)`` grid positions. Source-id RoPE (:mod:`.joyai_video_edit_rope`) is the only
signal separating them; drop it and the model cannot see what it is supposed to be editing, while
every shape still checks out.

**Modulation is AdaLN-single with a per-block learned offset.** ``condition_embedder`` emits one
shared ``[B, 6, 4096]`` time vector for the whole network; each block adds its own learned
``modulate_table (1, 6, 4096)`` to it before splitting out
``shift1, scale1, gate1, shift2, scale2, gate2``.

Upstream computes the three normalisation/RoPE steps with fused CUDA kernels from ``joyomni_ops``,
which -- unlike its attention path -- have **no** pure-torch fallback and raise when the package is
absent. The reference formulas therefore could not be read off upstream's source; they are derived in
``PORT_CONTRACT.md`` section 3 and implemented here as :func:`layernorm_modulate`,
:func:`qk_norm_rope` and :func:`add_gate`. These functions are the *contract* a later fused-kernel
step has to satisfy, which is why they are module-level and separately tested rather than inlined.

The three diffusers submodules (``FeedForward``, ``TimestepEmbedding``, ``PixArtAlphaTextProjection``)
are imported rather than re-implemented: the checkpoint's key names encode diffusers' own internals
(``img_mlp.net.0.proj.weight``, ``linear_1``/``linear_2``), so a local copy would have to reproduce
that naming anyway and would decouple nothing. ``test_transformer.py`` pins the constructed key set
instead, turning a future diffusers refactor into a fast CPU failure rather than a 30 GiB load error.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from einops import rearrange
from torch import nn

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    HEADS_NUM,
    HIDDEN_SIZE,
    IN_CHANNELS,
    MLP_WIDTH_RATIO,
    MM_DOUBLE_BLOCKS_DEPTH,
    NORM_EPS,
    NUM_MODULATION_CHUNKS,
    OUT_CHANNELS,
    PATCH_SIZE,
    ROPE_DIM_LIST,
    ROPE_THETA,
    SELF_ATTN_MODE_REF_IMAGE_CACHE,
    SOURCE_ID_EDIT_CONDITION,
    SOURCE_ID_EXTRA_REF_IMAGE,
    SOURCE_ID_ROPE_DIM,
    SOURCE_ID_ROPE_THETA,
    SOURCE_ID_TARGET,
    TEXT_STATES_DIM,
    TIME_FREQ_DIM,
)
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_kv import JoyKVWindow
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_rope import (
    apply_rotary_emb,
    compose_rope,
    generate_source_id_rope,
    get_rotary_pos_embed_from_ids,
    get_token_frame_ids,
)

# ``gelu-approximate`` selects diffusers' ``GELU(approximate="tanh")``, which is what puts the up
# projection at ``net.0.proj`` and the down projection at ``net.2`` in the checkpoint.
MLP_ACT_TYPE = "gelu-approximate"


# --- native replacements for the joyomni_ops fused kernels ---------------------------------------


def layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = NORM_EPS,
) -> torch.Tensor:
    """Affine-free LayerNorm followed by AdaLN modulation, computed in fp32.

    Replaces ``joyomni_ops.fused_norm_scale_shift``. ``shift``/``scale`` are ``[B, D]`` and broadcast
    over the sequence.

    The ``1 +`` on the scale is load-bearing and is not readable from upstream's source -- its wrapper
    hands the raw ``scale`` straight to the kernel, so the term is either inside the kernel or absent.
    It is inside: feeding the real kernel ``scale = shift = 0`` returns the plain LayerNorm rather than
    zeros, and this function's output then matches it bit for bit. Note the *gate* has no ``1 +`` (see
    :func:`add_gate`) -- the asymmetry is real and matches diffusers' Wan block.
    """
    if shift.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"`shift`/`scale` must be 2D [B, D], got {tuple(shift.shape)} / {tuple(scale.shape)}.")
    normed = F.layer_norm(x.float(), (x.shape[-1],), None, None, eps)
    return (normed * (1 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)).type_as(x)


def add_gate(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``residual + x * gate``, with the gate broadcast over the sequence.

    Upstream's ``fused_add_gate`` is already pure torch (``torch.addcmul``), so this is exact rather
    than a re-derivation. Kept as a named function because the missing ``1 +`` relative to
    :func:`layernorm_modulate`'s scale is the kind of asymmetry a reader "fixes".
    """
    return torch.addcmul(residual, x, gate.unsqueeze(1))


class RMSNorm(nn.Module):
    """Per-head RMSNorm over the last (head) dimension, weight only -- no bias.

    Upstream reaches this normalisation three different ways, and they are *not* numerically
    interchangeable, so this class exposes two methods rather than looking inconsistent by accident:

    * The text stream calls the module (``dit.py:379``), i.e. upstream's own torch ``RMSNorm``, which
      does ``_norm(x.float()).type_as(x) * weight`` -- rounding to bf16 *before* the weight multiply.
      :meth:`forward` reproduces that exactly.
    * The image q/k path and the KV-cache store call fused kernels, which keep the intermediate in
      registers and write bf16 exactly once. Rounding early costs 2.8e-3 relative against the real
      ``joyomni_ops.rmsnorm`` where :meth:`normalize_fp32` costs 7.7e-6.

    A reader "fixing" the asymmetry in either direction reintroduces a divergence in all 40 blocks.
    """

    def __init__(self, dim: int, eps: float = NORM_EPS, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def normalize_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """The fused kernels' semantics: one fp32 chain, left *uncast* so callers can chain onto it."""
        x32 = x.float()
        return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upstream's *torch* ``RMSNorm``: round to the input dtype before applying the weight."""
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm on q and k, **then** 3D RoPE on both.

    Replaces ``joyomni_ops.fused_qk_norm_rope_3d_paired``. Order matters: normalising after the
    rotation would rescale the rotated vectors and is not what the kernel does.

    The kernel is *fused*, so its post-norm intermediate never leaves registers. Going through
    :meth:`RMSNorm.forward` here would round it to bf16 and then immediately cast back up inside
    :func:`apply_rotary_emb` -- a rounding the kernel does not do, repeated in all 40 blocks. Chaining
    the fp32 value through instead measures 7.9e-6 against the kernel where the rounded version
    measures 3.3e-3.

    The rotary tables, however, go the *other* way, and this is the one asymmetry that a
    random-tensor probe cannot see. ``sgl_fused_ops.fused_qk_norm_rope_3d`` ends with
    ``cos_bf16 = cos.to(torch.bfloat16)`` and the kernel reads them back as
    ``__bfloat162float(cos_ptr[...])``, so upstream rotates by tables carrying only bf16's 8 mantissa
    bits. Applying the fp32 tables here is strictly *more* accurate and measurably not upstream:
    tapping (q, k, v) as passed to attention on a real frame, with every DiT input substituted from
    upstream's own dump, put ``v`` (a bare slice of the qkv projection) at 2.5e-05 and the text q/k
    (normed, never rotated) at 5.7e-05, while the rotated image q/k sat at **2.3e-03** -- uniform
    across all three of ``rope_dim_list``'s groups and a median of 2 bf16 ULPs, i.e. the signature of
    a precision difference rather than a wrong formula. 2**-9 is 2e-3, which is that number.

    Rounding the tables here and computing in fp32 reproduces the kernel exactly: bf16 in registers,
    converted to float, fp32 multiply-add, one bf16 store. Note the two *other* rope sites must not
    copy this -- upstream re-applies RoPE to cached pre-rope keys through its own torch
    ``apply_rotary_emb`` (``dit.py:209``) with the fp32 tables, and that function is character-for-
    character ours, so those paths already agree and rounding them would introduce a divergence.

    Both this path and the kernel take the full-width ``repeat_interleave(2)`` tables: the kernel's
    ``cos[..., ::2]`` is that operation's exact inverse, so the two conventions agree rather than one
    of them halving the frequency coverage.
    """
    cos, sin = freqs_cis
    # The deliberate round-trip: bf16 to drop the mantissa bits the kernel never sees, back to fp32 so
    # the multiply-add itself stays in the kernel's working precision. Relying on type promotion from
    # a left-over bf16 table would compute the same thing today and silently stop if a caller reorders
    # the operands, so the intent is spelled out.
    rounded = (cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float())
    return (
        apply_rotary_emb(q_norm.normalize_fp32(q), rounded).to(q.dtype),
        apply_rotary_emb(k_norm.normalize_fp32(k), rounded).to(k.dtype),
    )


def sdpa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Non-causal SDPA over ``[B, S, H, D]`` tensors.

    ``is_causal=False`` always: this model is chunk-autoregressive at the *rollout* level, and within
    a forward pass every token attends to every other one, including the concatenated text stream and
    the prepended KV history. A causal mask here would break the text fusion, not just the ordering.
    """
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=False
    )
    return out.transpose(1, 2)


@contextlib.contextmanager
def _module_factory_context(device, dtype) -> Iterator[None]:
    """Make submodules that take no ``device``/``dtype`` kwargs honour them anyway.

    ``FeedForward`` holds two thirds of every block's parameters, so constructing it at the default
    fp32 and casting afterwards would transiently need ~4x the model's final footprint (65 GiB for
    this checkpoint). Upstream does exactly that and then blanket-casts; the end state is identical
    because the checkpoint is uniformly bf16, but the peak is not.
    """
    with contextlib.ExitStack() as stack:
        if device is not None:
            stack.enter_context(torch.device(device))
        if dtype is not None:
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            stack.callback(torch.set_default_dtype, previous_dtype)
        yield


def _broadcast_freqs(freqs: tuple[torch.Tensor, torch.Tensor], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Give ``[L, D]`` rotary tables a leading batch dim that :func:`apply_rotary_emb` can consume.

    ``reshape_for_broadcast`` uses ``view``, not broadcasting, so the tables must already carry the
    real batch size -- and ``expand`` alone leaves a stride-0 dim that ``view`` rejects. Materialised
    with ``repeat`` for ``batch_size > 1``; the single-request path (the only one the MVP pipeline
    admits) pays nothing.
    """
    if batch_size == 1:
        return freqs[0].unsqueeze(0), freqs[1].unsqueeze(0)
    return freqs[0].unsqueeze(0).repeat(batch_size, 1, 1), freqs[1].unsqueeze(0).repeat(batch_size, 1, 1)


def _row_shared_position_ids(
    ids: torch.Tensor | Sequence[int],
    *,
    name: str,
    device: torch.device,
    expected_len: int | None = None,
) -> torch.Tensor:
    """Normalise ``[L]`` or ``[B, L]`` position ids to the single 1D row the RoPE tables are built from.

    One rotary table is built per forward and shared by the whole batch, so per-sample positions
    cannot be honoured. Upstream takes ``ids[0]`` and silently discards rows 1..N; this raises
    instead, which is the earliest point the "batch > 1 loses samples" failure can be caught.
    """
    tensor = torch.as_tensor(ids, device=device, dtype=torch.long)
    if tensor.ndim == 2:
        if tensor.shape[0] > 1 and not torch.equal(tensor, tensor[:1].expand_as(tensor)):
            raise ValueError(
                f"`{name}` differs across the batch; a single shared rotary table cannot represent "
                f"per-sample positions. Run one request per forward."
            )
        tensor = tensor[0]
    elif tensor.ndim != 1:
        raise ValueError(f"`{name}` must be 1D [L] or 2D [B, L], got shape {tuple(tensor.shape)}.")
    if expected_len is not None and tensor.numel() != expected_len:
        raise ValueError(f"`{name}` must have {expected_len} entries, got {tensor.numel()}.")
    return tensor


# --- modules --------------------------------------------------------------------------------------


class ModulateWan(nn.Module):
    """AdaLN-single modulation: a learned per-block offset on the network-wide time vector.

    ``modulate_table`` is the only per-block conditioning parameter. Splitting the sum into
    ``factor`` chunks along dim 1 yields, in order,
    ``shift1, scale1, gate1, shift2, scale2, gate2``.
    """

    def __init__(self, hidden_size: int, factor: int, device=None, dtype=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.randn(1, factor, hidden_size, device=device, dtype=dtype) / hidden_size**0.5
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.ndim != 3:
            x = x.unsqueeze(1)
        return [chunk.squeeze(1) for chunk in (self.modulate_table + x).chunk(self.factor, dim=1)]


class WanTimeTextImageEmbedding(nn.Module):
    """Shared conditioning: a 6-way time vector plus the projected text embeddings.

    ``time_proj`` emits ``6 * hidden`` in one Linear, which the caller unflattens to ``[B, 6, hidden]``.
    """

    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_shape = timestep.shape
        timestep_is_sequence = timestep.ndim > 1
        if timestep_is_sequence:
            timestep = timestep.flatten()

        timestep = self.timesteps_proj(timestep)
        embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != embedder_dtype:
            timestep = timestep.to(embedder_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        if timestep_is_sequence:
            timestep_proj = timestep_proj.reshape(*timestep_shape, timestep_proj.shape[-1])

        return timestep_proj, self.text_embedder(encoder_hidden_states)


class MMDoubleStreamBlock(nn.Module):
    """One dual-stream block: parallel image/text streams joined by a single self-attention.

    The two streams keep separate weights throughout (``img_*`` / ``txt_*``) and are concatenated
    only to compute attention, then split apart again. ``skip_text_stream`` short-circuits the text
    half entirely, which is what the clean-latent KV-store forward uses.
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = MLP_ACT_TYPE,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        for stream in ("img", "txt"):
            setattr(self, f"{stream}_mod", ModulateWan(hidden_size, NUM_MODULATION_CHUNKS, **factory_kwargs))
            # Affine-free: modulation supplies the scale and shift, so these carry no checkpoint keys.
            setattr(
                self,
                f"{stream}_norm1",
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs),
            )
            setattr(self, f"{stream}_attn_qkv", nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs))
            setattr(self, f"{stream}_attn_q_norm", RMSNorm(head_dim, eps=NORM_EPS, **factory_kwargs))
            setattr(self, f"{stream}_attn_k_norm", RMSNorm(head_dim, eps=NORM_EPS, **factory_kwargs))
            setattr(self, f"{stream}_attn_proj", nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs))
            setattr(
                self,
                f"{stream}_norm2",
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=NORM_EPS, **factory_kwargs),
            )
            setattr(
                self, f"{stream}_mlp", FeedForward(hidden_size, inner_dim=mlp_hidden_dim, activation_fn=mlp_act_type)
            )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vis_freqs_cis: tuple[torch.Tensor, torch.Tensor],
        *,
        layer_idx: int | None = None,
        kv_window: JoyKVWindow | None = None,
        skip_text_stream: bool = False,
        kv_cache_pre_rope: bool = False,
        cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        memoize_kv_assembly: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reads = kv_window is not None and kv_window.reads
        writes = kv_window is not None and kv_window.writes

        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(vec)

        img_modulated = layernorm_modulate(img, img_shift1, img_scale1, self.img_norm1.eps)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Pre-RoPE caching: store the key normed but *un-rotated*, because the window renumbers
        # positions every chunk (see joyai_video_edit_kv). Only the writer consumes this, so it is
        # skipped on read-only forwards -- 40 layers of RMSNorm over the whole image stream.
        # `normalize_fp32`, not `forward`: upstream reaches this through the `rmsnorm_qk_bf16` kernel
        # (dit.py:356), whose single bf16 write `forward`'s early round would not reproduce.
        img_k_for_cache = None
        if writes and kv_cache_pre_rope:
            img_k_for_cache = self.img_attn_k_norm.normalize_fp32(img_k).to(img_k.dtype)

        img_q, img_k = qk_norm_rope(img_q, img_k, self.img_attn_q_norm, self.img_attn_k_norm, vis_freqs_cis)
        if writes and not kv_cache_pre_rope:
            img_k_for_cache = img_k

        if skip_text_stream:
            query, key, value = img_q, img_k, img_v
        else:
            txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(vec)
            txt_modulated = layernorm_modulate(txt, txt_shift1, txt_scale1, self.txt_norm1.eps)
            txt_qkv = self.txt_attn_qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            # No RoPE on the text stream: its positions already come from the MLLM encoder.
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v.dtype)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v.dtype)
            # Image first, text second -- the split below and the cache write both depend on it.
            query = torch.cat((img_q, txt_q), dim=1)
            key = torch.cat((img_k, txt_k), dim=1)
            value = torch.cat((img_v, txt_v), dim=1)

        if writes:
            # Only the image stream is ever cached; text keys are recomputed every chunk.
            kv_window.write(layer_idx, img_k_for_cache, img_v)

        if reads:
            cached_key, cached_value = kv_window.assemble(
                layer_idx,
                device=query.device,
                dtype=query.dtype,
                cached_freqs_cis=cached_freqs_cis if kv_cache_pre_rope else None,
                memoize=memoize_kv_assembly,
            )
        else:
            cached_key = cached_value = None

        if cached_key is not None:
            # Prepended, giving key layout [cached | img | txt] against queries [img | txt]. The
            # query is deliberately not extended: history is attended to, never denoised.
            key = torch.cat([cached_key, key], dim=1)
            value = torch.cat([cached_value, value], dim=1)

        attn = sdpa_attention(query, key, value).flatten(2, 3)
        if skip_text_stream:
            img_attn = attn
        else:
            # Split at img.shape[1], which includes the reference-video tokens.
            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        img = add_gate(img, self.img_attn_proj(img_attn), img_gate1)
        img = add_gate(
            img, self.img_mlp(layernorm_modulate(img, img_shift2, img_scale2, self.img_norm2.eps)), img_gate2
        )

        if not skip_text_stream:
            txt = add_gate(txt, self.txt_attn_proj(txt_attn), txt_gate1)
            txt_modulated2 = layernorm_modulate(txt, txt_shift2, txt_scale2, self.txt_norm2.eps)
            txt = add_gate(txt, self.txt_mlp(txt_modulated2), txt_gate2)

        return img, txt


class JoyAIVideoEditTransformer3DModel(ModelMixin, ConfigMixin):
    """The 40-block MMDiT denoiser.

    Defaults are the shipped checkpoint's values, not upstream's ``Transformer3DModel`` defaults
    (which describe a different, smaller model). ``patch_size`` is ``[1, 1, 1]``: this DiT performs no
    patchification at all, since the VAE has already compressed 24x spatially.

    KV state lives in a caller-owned :class:`JoyKVWindow` rather than on the module, so a request's
    cache is not smuggled through module attributes and a later streaming phase can swap the backing
    store without touching the blocks.
    """

    _supports_gradient_checkpointing = False
    ignore_for_config = ["device", "dtype"]

    @register_to_config
    def __init__(
        self,
        patch_size: Sequence[int] | None = None,
        in_channels: int = IN_CHANNELS,
        out_channels: int | None = OUT_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        heads_num: int = HEADS_NUM,
        text_states_dim: int = TEXT_STATES_DIM,
        mlp_width_ratio: float = MLP_WIDTH_RATIO,
        mm_double_blocks_depth: int = MM_DOUBLE_BLOCKS_DEPTH,
        rope_dim_list: Sequence[int] | None = None,
        theta: float = ROPE_THETA,
        chunk_size: int | None = None,
        local_window_size: int = 3,
        global_sink_chunk: bool = True,
        causal: bool = False,
        source_id_rope_dim: int = SOURCE_ID_ROPE_DIM,
        source_id_rope_theta: float = SOURCE_ID_ROPE_THETA,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.patch_size = list(PATCH_SIZE if patch_size is None else patch_size)
        self.rope_dim_list = list(ROPE_DIM_LIST if rope_dim_list is None else rope_dim_list)
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.theta = theta
        self.chunk_size = chunk_size
        self.local_window_size = local_window_size
        self.global_sink_chunk = global_sink_chunk
        # A declaration about the *rollout*, not an attention mask, and the name invites the opposite
        # reading. Upstream's DiT reads it only here (to require `chunk_size`); attention is
        # unconditionally non-causal at both call sites. The consumers all sit outside the DiT: the
        # pipeline gates its sliding-window VAE encode on it and resolves `chunk_size` from it, and the
        # streaming session refuses a config without it. It is `True` in the shipped deployment config,
        # so "chunk-autoregressive with a bounded KV window" -- which is what this port implements --
        # is the production path, and a port that refused the flag would refuse the real config.
        self.causal = causal
        self.source_id_rope_dim = int(source_id_rope_dim)
        self.source_id_rope_theta = float(source_id_rope_theta)

        if hidden_size % heads_num != 0:
            raise ValueError(f"`hidden_size` {hidden_size} must be divisible by `heads_num` {heads_num}.")
        head_dim = hidden_size // heads_num
        if sum(self.rope_dim_list) != head_dim:
            raise ValueError(f"sum(rope_dim_list)={sum(self.rope_dim_list)} must equal head_dim={head_dim}.")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive when provided, got {chunk_size}.")
        if local_window_size <= 0:
            raise ValueError(f"`local_window_size` must be positive, got {local_window_size}.")
        # `causal` names the *rollout*, not an attention mask -- see the note on `self.causal` below.
        if causal and chunk_size is None:
            raise ValueError("`chunk_size` must be provided when `causal=True`.")

        with _module_factory_context(device, dtype):
            self.img_in = nn.Conv3d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
            self.condition_embedder = WanTimeTextImageEmbedding(
                dim=hidden_size,
                time_freq_dim=TIME_FREQ_DIM,
                time_proj_dim=hidden_size * NUM_MODULATION_CHUNKS,
                text_embed_dim=text_states_dim,
            )
            self.double_blocks = nn.ModuleList(
                [
                    MMDoubleStreamBlock(hidden_size, heads_num, mlp_width_ratio=mlp_width_ratio)
                    for _ in range(mm_double_blocks_depth)
                ]
            )
            # Parameter-free, and there is no final AdaLN: the head is literally proj_out(norm_out(x)).
            self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=NORM_EPS)
            self.proj_out = nn.Linear(hidden_size, self.out_channels * math.prod(self.patch_size))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.heads_num

    def _patch_shape(self, latent: torch.Tensor) -> tuple[int, int, int]:
        _, _, num_frames, height, width = latent.shape
        pt, ph, pw = self.patch_size
        return num_frames // pt, height // ph, width // pw

    def _rotary_for(self, frame_ids: torch.Tensor, spatial_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        return get_rotary_pos_embed_from_ids(
            frame_ids=frame_ids,
            spatial_shape=spatial_shape,
            head_dim=self.head_dim,
            rope_dim_list=self.rope_dim_list,
            theta=self.theta,
        )

    def _cached_rotary(
        self,
        cached_temporal_ids: torch.Tensor | Sequence[int],
        spatial_shape: tuple[int, int],
        device: torch.device,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """3D RoPE tables for the cached history, in window order.

        The history's spatial grid is by definition the current chunk's, so ``spatial_shape`` is
        reused; only the temporal ids differ, and they are the *renumbered* window positions rather
        than the frames' original indices. No source-id rotation is composed in: cached tokens are
        always clean history, i.e. ``source_id == 0``, which is the identity.
        """
        cached_ids = _row_shared_position_ids(cached_temporal_ids, name="cached_temporal_ids", device=device)
        cached_frame_ids = get_token_frame_ids(
            (cached_ids.numel(), spatial_shape[0], spatial_shape[1]), device, temporal_ids=cached_ids
        )
        return _broadcast_freqs(self._rotary_for(cached_frame_ids, spatial_shape), batch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        ref_video_latent: torch.Tensor | None = None,
        current_temporal_ids: torch.Tensor | Sequence[int] | None = None,
        cached_temporal_ids: torch.Tensor | Sequence[int] | None = None,
        kv_window: JoyKVWindow | None = None,
        kv_cache_mode: str | None = None,
        kv_cache_scope: str | None = None,
        kv_cache_chunk_id: int | None = None,
        kv_cache_selected_chunk_ids: list[int] | None = None,
        kv_cache_pre_rope: bool = False,
        self_attn_input_mode: str | None = None,
        skip_text_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise one latent chunk.

        ``encoder_hidden_states_mask`` is accepted and **ignored**. Upstream takes it, casts it to
        bool, and never reads it again -- text is attended to unmasked as part of the concatenated
        self-attention. It stays in the signature so callers written against upstream keep working,
        and ``test_transformer.py`` asserts two different masks give identical output so this claim
        cannot rot.

        ``ref_video_latent`` is the clean source chunk. It is appended to the image stream and shares
        the noisy chunk's temporal ids, distinguished only by its source id.
        """
        del encoder_hidden_states_mask  # documented above: required by upstream's signature, unused

        if kv_window is not None:
            kv_window.configure(
                scope=kv_cache_scope,
                mode=kv_cache_mode,
                chunk_id=kv_cache_chunk_id,
                selected_chunk_ids=kv_cache_selected_chunk_ids,
                pre_rope=kv_cache_pre_rope,
            )

        device = hidden_states.device
        batch_size = hidden_states.shape[0]
        current_patch_shape = self._patch_shape(hidden_states)
        current_seq_len = math.prod(current_patch_shape)
        spatial_shape = (current_patch_shape[1], current_patch_shape[2])

        temporal_ids = None
        if current_temporal_ids is not None:
            temporal_ids = _row_shared_position_ids(
                current_temporal_ids,
                name="current_temporal_ids",
                device=device,
                expected_len=current_patch_shape[0],
            )

        hidden_tokens = self.img_in(hidden_states).flatten(2).transpose(1, 2)
        current_frame_ids = get_token_frame_ids(current_patch_shape, device, temporal_ids=temporal_ids)

        # The whole chunk carries one source id. Prefilling the static reference *image* marks it as
        # an extra reference; everything else is the target being denoised.
        target_source_id = (
            SOURCE_ID_EXTRA_REF_IMAGE if self_attn_input_mode == SELF_ATTN_MODE_REF_IMAGE_CACHE else SOURCE_ID_TARGET
        )
        latent_segments = [hidden_tokens]
        rotary_segments = [self._rotary_for(current_frame_ids, spatial_shape)]
        source_id_segments = [torch.full((current_seq_len,), target_source_id, device=device, dtype=torch.float32)]

        if ref_video_latent is not None:
            if ref_video_latent.shape[0] != batch_size:
                raise ValueError(
                    f"`ref_video_latent` batch {ref_video_latent.shape[0]} != hidden_states batch {batch_size}."
                )
            ref_patch_shape = self._patch_shape(ref_video_latent)
            if ref_patch_shape[1:] != current_patch_shape[1:]:
                raise ValueError(
                    "`ref_video_latent` spatial patch shape must match the noisy latent's: "
                    f"{ref_patch_shape[1:]} != {current_patch_shape[1:]}."
                )
            ref_tokens = self.img_in(ref_video_latent).flatten(2).transpose(1, 2)
            # Same temporal ids as the target on purpose: source and target overlay each other in
            # position space and are separated only by the source id below.
            ref_frame_ids = get_token_frame_ids(ref_patch_shape, device, temporal_ids=temporal_ids)
            latent_segments.append(ref_tokens)
            rotary_segments.append(self._rotary_for(ref_frame_ids, spatial_shape))
            source_id_segments.append(
                torch.full((ref_tokens.shape[1],), SOURCE_ID_EDIT_CONDITION, device=device, dtype=torch.float32)
            )

        img = torch.cat(latent_segments, dim=1)
        freqs_3d = _broadcast_freqs(
            (
                torch.cat([cos for cos, _ in rotary_segments], dim=0),
                torch.cat([sin for _, sin in rotary_segments], dim=0),
            ),
            batch_size,
        )
        visual_source_id = torch.cat(source_id_segments, dim=0).unsqueeze(0)
        freqs_role = generate_source_id_rope(
            source_id=visual_source_id,
            head_dim=self.head_dim,
            device=freqs_3d[0].device,
            dtype=freqs_3d[0].dtype,
            source_id_rope_dim=self.source_id_rope_dim,
            source_id_rope_theta=self.source_id_rope_theta,
        )
        vis_freqs_cis = compose_rope(freqs_3d, freqs_role)

        vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(-1, (NUM_MODULATION_CHUNKS, -1))

        # Rotating cached keys needs a position table too. Both denoise steps of a chunk see the same
        # window and the same positions, so it is computed once and memoised on the window.
        reads = kv_window is not None and kv_window.reads
        memoize = bool(reads and kv_cache_pre_rope and cached_temporal_ids is not None)
        cached_freqs_cis = kv_window.memo_lookup(cached_temporal_ids) if memoize else None
        if cached_freqs_cis is None and kv_cache_pre_rope and cached_temporal_ids is not None:
            cached_freqs_cis = self._cached_rotary(cached_temporal_ids, spatial_shape, device, batch_size)
            if memoize:
                kv_window.memo_store(cached_temporal_ids, cached_freqs_cis)

        for layer_idx, block in enumerate(self.double_blocks):
            img, txt = block(
                img,
                txt,
                vec,
                vis_freqs_cis,
                layer_idx=layer_idx,
                kv_window=kv_window,
                skip_text_stream=skip_text_stream,
                kv_cache_pre_rope=kv_cache_pre_rope,
                cached_freqs_cis=cached_freqs_cis,
                memoize_kv_assembly=memoize,
            )

        img = self.proj_out(self.norm_out(img))
        # Reference tokens were appended, so the target is the leading slice; they are dropped here
        # rather than being decoded.
        img = img[:, :current_seq_len]
        return self.unpatchify(img, *current_patch_shape), txt

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """Tokens back to a ``[B, C, T, H, W]`` latent. A reshape at ``patch_size == [1, 1, 1]``."""
        channels = self.out_channels
        pt, ph, pw = self.patch_size
        if t * h * w != x.shape[1]:
            raise ValueError(f"Token count {x.shape[1]} does not match grid {(t, h, w)}.")
        x = x.reshape(x.shape[0], t, h, w, channels, pt, ph, pw)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        return x.reshape(x.shape[0], channels, t * pt, h * ph, w * pw)
