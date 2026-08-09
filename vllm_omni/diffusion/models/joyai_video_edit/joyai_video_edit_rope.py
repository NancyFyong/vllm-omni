# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""Rotary embeddings for JoyAI-Video-Edit.

Two rotations are composed:

1. **3D RoPE** over ``(t, h, w)`` with per-axis dims ``[16, 56, 56]`` at ``theta=256``.
2. **Source-id RoPE**, keyed by which stream a token came from.

The second one is load-bearing rather than cosmetic. The reference (source) video latent is given the
*same* temporal ids as the noisy chunk being denoised, so source and target tokens land on identical
``(t, h, w)`` positions; ``source_id`` is the only signal separating them. ``source_id == 0`` yields
``cos=1, sin=0``, so target tokens are left unrotated by this second stage.

Layout note: this is the interleaved (GPT-J) convention -- pairs are ``(x[0], x[1])``, ``(x[2], x[3])``,
... -- not the half-split (NeoX) convention. ``cos``/``sin`` are therefore full head-dim width, built
with ``repeat_interleave(2)``. Tensors are ``[B, L, H, D]``, not the ``[B, H, L, D]`` that SDPA wants.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    ROPE_DIM_LIST,
    ROPE_THETA,
    SOURCE_ID_ROPE_DIM,
    SOURCE_ID_ROPE_THETA,
)


def reshape_for_broadcast(
    freqs_cis: tuple[torch.Tensor, torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """View ``(cos, sin)`` as ``[B, L, 1, D]`` so they broadcast over heads of an ``[B, L, H, D]`` ``x``."""
    ndim = x.ndim
    shape = [d if i in (0, 1, ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved rotation partner: ``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``."""
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply full-width interleaved RoPE to ``x`` of shape ``[B, L, H, D]``."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, got {x.shape[-1]}")
    cos, sin = reshape_for_broadcast(freqs_cis, x)
    cos, sin = cos.to(x.device), sin.to(x.device)
    return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor | int,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables of width ``dim`` for the given positions, interleaved via ``repeat_interleave(2)``."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    pos_device = pos.device if torch.is_tensor(pos) else torch.device("cpu")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos_device)[: (dim // 2)].float() / dim))
    freqs = torch.outer(pos * interpolation_factor, freqs)
    return freqs.cos().repeat_interleave(2, dim=1), freqs.sin().repeat_interleave(2, dim=1)


def get_token_frame_ids(
    post_patch_shape: tuple[int, int, int],
    device: torch.device,
    temporal_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token frame index for a ``[T, H, W]`` grid flattened frame-major.

    Token order must match ``img_in(x).flatten(2).transpose(1, 2)``, i.e. all tokens of frame 0, then
    frame 1, ... -- hence ``repeat_interleave`` rather than ``repeat``.

    ``temporal_ids`` overrides the default ``arange(T)`` so a chunk can be placed at renumbered
    positions inside the KV window.
    """
    num_frames, post_patch_height, post_patch_width = post_patch_shape
    spatial_tokens_per_frame = post_patch_height * post_patch_width
    if temporal_ids is None:
        frame_ids = torch.arange(num_frames, device=device, dtype=torch.long)
    else:
        frame_ids = torch.as_tensor(temporal_ids, device=device, dtype=torch.long)
        if frame_ids.ndim != 1 or frame_ids.numel() != num_frames:
            raise ValueError(f"`temporal_ids` must be 1D with length {num_frames}, got {tuple(frame_ids.shape)}.")
    return frame_ids.repeat_interleave(spatial_tokens_per_frame)


def get_rotary_pos_embed_from_ids(
    *,
    frame_ids: torch.Tensor,
    spatial_shape: tuple[int, int],
    head_dim: int,
    rope_dim_list: list[int] = ROPE_DIM_LIST,
    theta: float = ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE tables for tokens whose frame index is given per token.

    ``frame_ids`` is already expanded to one entry per token (see :func:`get_token_frame_ids`); the
    ``h``/``w`` positions are reconstructed from ``spatial_shape`` and tiled across frames.
    """
    post_patch_height, post_patch_width = spatial_shape
    device = frame_ids.device
    temporal_positions = frame_ids.to(dtype=torch.float32)
    spatial_tokens_per_frame = post_patch_height * post_patch_width
    if temporal_positions.numel() % spatial_tokens_per_frame != 0:
        raise ValueError(
            f"`frame_ids` length {temporal_positions.numel()} is not divisible by "
            f"spatial token count {spatial_tokens_per_frame}."
        )
    if sum(rope_dim_list) != head_dim:
        raise ValueError(f"sum(rope_dim_list)={sum(rope_dim_list)} must equal head_dim={head_dim}")

    num_frames = temporal_positions.numel() // spatial_tokens_per_frame
    h_positions = torch.arange(post_patch_height, dtype=torch.float32, device=device)
    w_positions = torch.arange(post_patch_width, dtype=torch.float32, device=device)
    h_grid, w_grid = torch.meshgrid(h_positions, w_positions, indexing="ij")
    h_positions = h_grid.reshape(-1).repeat(num_frames)
    w_positions = w_grid.reshape(-1).repeat(num_frames)

    cos_list, sin_list = [], []
    for dim, positions in zip(rope_dim_list, (temporal_positions, h_positions, w_positions), strict=True):
        cos, sin = get_1d_rotary_pos_embed(dim, positions, theta=theta)
        cos_list.append(cos)
        sin_list.append(sin)
    return torch.cat(cos_list, dim=1), torch.cat(sin_list, dim=1)


def generate_source_id_rope(
    source_id: torch.Tensor,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    source_id_rope_dim: int = SOURCE_ID_ROPE_DIM,
    source_id_rope_theta: float = SOURCE_ID_ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables encoding a token's provenance (target / edit condition / extra reference).

    Slots beyond ``source_id_rope_dim`` stay at ``cos=1, sin=0`` (identity), so a ``source_id`` of 0
    leaves the 3D rotation untouched.
    """
    role_dim = max(0, min(int(source_id_rope_dim), int(head_dim)))
    half_head = head_dim // 2

    cos_half = torch.ones(*source_id.shape, half_head, device=device, dtype=torch.float32)
    sin_half = torch.zeros(*source_id.shape, half_head, device=device, dtype=torch.float32)

    inv_freq = 1.0 / (
        source_id_rope_theta ** (torch.arange(0, role_dim, 2, device=device, dtype=torch.float32) / role_dim)
    )
    angles = source_id.unsqueeze(-1) * inv_freq
    cos_half[..., : role_dim // 2] = torch.cos(angles)
    sin_half[..., : role_dim // 2] = torch.sin(angles)

    return (
        cos_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
        sin_half.repeat_interleave(2, dim=-1).to(dtype=dtype),
    )


def compose_rope(
    freqs_3d: tuple[torch.Tensor, torch.Tensor],
    freqs_role: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose two rotations into one.

    Complex multiplication of the two unit phasors, so applying the result once is equivalent to
    rotating by the 3D angle and then by the source-id angle.
    """
    cos_3d, sin_3d = freqs_3d
    cos_role, sin_role = freqs_role
    return (
        cos_3d * cos_role - sin_3d * sin_role,
        sin_3d * cos_role + cos_3d * sin_role,
    )
