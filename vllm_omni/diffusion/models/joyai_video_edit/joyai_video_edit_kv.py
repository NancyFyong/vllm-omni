# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jd-opensource/JoyAI-Video-Edit
"""Bounded sliding KV window for JoyAI-Video-Edit's chunked autoregressive rollout.

The model generates one latent chunk at a time and attends to a *bounded* set of earlier chunks:
chunk 0 (a global "sink") plus the two most recent. That bound is what keeps both memory and the
temporal position range constant no matter how long the video is.

Two details make this window unlike a conventional paged KV cache, and are why it lives here rather
than on top of :mod:`vllm_omni.experimental.ar_diffusion`:

**Keys are stored pre-RoPE.** Window positions are renumbered contiguously every chunk, so a cached
token's temporal index *changes* from one chunk to the next. Keys are therefore stored RMSNorm'd but
un-rotated, and rotated at read time using the current chunk's ``cached_temporal_ids``. Storing
post-RoPE keys would mean rewriting every resident entry each chunk.

**Only the image stream is cached.** Text is fused by concatenation into self-attention rather than
cross-attention, and text keys are never cached; cached keys are *prepended*, so the assembled key
layout is ``[cached | img | txt]`` while queries cover only ``[img | txt]``.

The class is deliberately weight-free and holds no module state, so a later streaming phase can
re-back it with a paged pool without touching the DiT blocks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import KV_CACHE_ID_REF_IMAGE
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_rope import apply_rotary_emb

KVEntry = dict[str, Any]

# Cache modes, matching upstream's strings.
MODE_REUSE = "reuse"  # read only
MODE_STORE = "store"  # write only
MODE_REUSE_STORE = "reuse_store"  # both
_CACHE_MODES = frozenset({MODE_REUSE, MODE_STORE, MODE_REUSE_STORE})


def kv_cache_memory_id(kind: str, chunk_id: int | None = None) -> int:
    """Map a cache slot to its integer key.

    Clean latent chunks use their own index; the static reference image uses a negative sentinel so
    it can never collide with a chunk id.
    """
    if kind == "clean":
        if chunk_id is None:
            raise ValueError("`chunk_id` is required for kind='clean'.")
        return int(chunk_id)
    if kind == "ref_image":
        return KV_CACHE_ID_REF_IMAGE
    raise ValueError(f"Unsupported cache kind: {kind!r}")


def chunk_frame_bounds(chunk_id: int, chunk_size: int, total_latent_frames: int) -> tuple[int, int]:
    chunk_start = chunk_id * chunk_size
    return chunk_start, min(total_latent_frames, chunk_start + chunk_size)


def get_chunk_windows(
    total_latent_frames: int,
    chunk_size: int,
    window_size: int,
    global_sink_chunk: bool,
) -> list[dict[str, Any]]:
    """Per-chunk attention windows for a rollout of ``total_latent_frames`` latent frames.

    With ``global_sink_chunk`` the window is chunk 0 plus the ``window_size - 1`` most recent chunks,
    so it never exceeds ``window_size`` entries: ``[0]``, ``[0, 1]``, ``[0, 1, 2]``, ``[0, 2, 3]``,
    ``[0, 3, 4]``, ...
    """
    if window_size <= 0:
        raise ValueError(f"`window_size` must be positive, got {window_size}.")

    windows = []
    num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        chunk_start, chunk_end = chunk_frame_bounds(chunk_idx, chunk_size, total_latent_frames)
        if global_sink_chunk and chunk_idx > 0:
            tail_window_size = max(window_size - 1, 1)
            tail_chunk_start = max(1, chunk_idx - tail_window_size + 1)
            selected_chunk_ids = [0] + list(range(tail_chunk_start, chunk_idx + 1))
        else:
            window_chunk_start = max(0, chunk_idx - window_size + 1)
            selected_chunk_ids = list(range(window_chunk_start, chunk_idx + 1))

        windows.append(
            {
                "chunk_idx": chunk_idx,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "selected_chunk_ids": selected_chunk_ids,
            }
        )
    return windows


def gather_window_temporal_ids(
    selected_chunk_ids: list[int],
    chunk_size: int,
    total_latent_frames: int,
    device: torch.device,
    max_temporal_ids: int | None = None,
) -> torch.Tensor:
    """Temporal position ids for every latent frame in the window, in window order.

    Default (``max_temporal_ids is None``): positions are **renumbered contiguously** from 0, so a
    window of chunks ``[0, 3, 4]`` becomes positions ``[0, 1, 2]``. This is what keeps the position
    range bounded, and it is why keys must be cached pre-RoPE.

    When ``max_temporal_ids`` is set, absolute frame indices are kept but shifted down to fit under
    that ceiling instead. Upstream leaves this unset, so the contiguous branch is the live one.
    """
    if max_temporal_ids is not None:
        abs_ids = torch.cat(
            [
                torch.arange(*chunk_frame_bounds(cid, chunk_size, total_latent_frames), device=device, dtype=torch.long)
                for cid in selected_chunk_ids
            ],
            dim=0,
        )
        shift = (abs_ids.max() - int(max_temporal_ids)).clamp_min(0)
        return (abs_ids - shift).clamp_min(0)

    temporal_ids = []
    offset = 0
    for cid in selected_chunk_ids:
        frame_start, frame_end = chunk_frame_bounds(cid, chunk_size, total_latent_frames)
        chunk_len = frame_end - frame_start
        temporal_ids.append(torch.arange(offset, offset + chunk_len, device=device, dtype=torch.long))
        offset += chunk_len
    return torch.cat(temporal_ids, dim=0)


def split_window_temporal_ids(window_ids: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Split window positions into ``(cached, current)``.

    The window's position table covers history *and* the active chunk. The trailing ``chunk_size``
    entries are the active chunk's own positions; everything before them belongs to the cached
    history, in window order. Returns ``None`` for ``cached`` on the first chunk, where there is no
    history.

    This split is the alignment contract between the position table and the cache: the assembled key
    is ``[cached | img | txt]``, and ``cached`` must line up with exactly the history entries.
    Handing the *full* table to the cache instead of this prefix shifts every cached key by
    ``chunk_size`` positions -- output stays finite and plausible while error accumulates with chunk
    index, which is why this is a function with a test rather than a slice at the call site.
    """
    if window_ids.ndim != 1:
        raise ValueError(f"`window_ids` must be 1D, got shape {tuple(window_ids.shape)}.")
    if chunk_size <= 0:
        raise ValueError(f"`chunk_size` must be positive, got {chunk_size}.")
    if window_ids.numel() < chunk_size:
        raise ValueError(f"`window_ids` has {window_ids.numel()} entries, fewer than chunk_size={chunk_size}.")
    current = window_ids[-chunk_size:]
    cached = window_ids[:-chunk_size]
    return (cached if cached.numel() > 0 else None), current


def cache_memory_ids_for_read(history_chunk_ids: Iterable[int], *, has_ref_image: bool) -> list[int]:
    """Window read order: history chunks first, then the reference image.

    The reference entry is **appended**, so the history entries occupy the leading positions of the
    cached table and stay aligned with it regardless of whether a reference image is present.
    """
    ids = [kv_cache_memory_id("clean", cid) for cid in history_chunk_ids]
    if has_ref_image:
        ids.append(kv_cache_memory_id("ref_image"))
    return ids


def next_selected_chunk_ids(
    chunk_idx: int,
    chunk_size: int,
    window_size: int,
    global_sink_chunk: bool,
) -> list[int]:
    """The window the *next* chunk will need -- i.e. what must survive eviction after a store."""
    windows = get_chunk_windows(
        total_latent_frames=chunk_idx + 2,
        chunk_size=chunk_size,
        window_size=window_size,
        global_sink_chunk=global_sink_chunk,
    )
    return list(windows[-1]["selected_chunk_ids"])


def concat_kv_entries(
    entries: Iterable[KVEntry],
    *,
    device: torch.device,
    dtype: torch.dtype,
    cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Concatenate window entries into one key/value pair, rotating pre-RoPE keys on the way.

    ``cached_freqs_cis`` covers the cached history in window order, so each pre-RoPE entry consumes
    the next ``seq_len`` slice. Entries stored *post*-RoPE -- the static reference image -- are passed
    through untouched and do not advance the offset. Rotating them again is the bug this guards
    against; the non-advancing part is defensive, since production appends the reference entry last
    (see :func:`cache_memory_ids_for_read`) so nothing follows it to misalign.
    """
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    pre_rope_offset = 0

    for entry in entries:
        if entry is None:
            continue
        key, value = entry.get("key"), entry.get("value")
        if key is None or value is None:
            continue

        key = key.to(device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)

        if entry.get("pre_rope", False) and cached_freqs_cis is not None:
            cos_all, sin_all = cached_freqs_cis
            seg_len = key.shape[1]
            cos_seg = cos_all[..., pre_rope_offset : pre_rope_offset + seg_len, :]
            sin_seg = sin_all[..., pre_rope_offset : pre_rope_offset + seg_len, :]
            key = apply_rotary_emb(key, (cos_seg, sin_seg))
            pre_rope_offset += seg_len

        keys.append(key)
        values.append(value)

    if not keys:
        return None, None
    return torch.cat(keys, dim=1), torch.cat(values, dim=1)


class JoyKVWindow:
    """Per-request KV store for the bounded chunk window.

    Layout is ``store[scope][chunk_id][layer_idx] -> {"key", "value", "pre_rope"}``. A forward pass
    first calls :meth:`configure` to declare what it intends to read and write, then the DiT blocks
    call :meth:`read` / :meth:`assemble` / :meth:`write` per layer.
    """

    def __init__(self, scopes: tuple[str, ...] = ("cond", "uncond")):
        self._scopes = scopes
        self._store: dict[str, dict[int, dict[int, KVEntry]]] = {s: {} for s in scopes}
        self.reset()

    # -- lifecycle ------------------------------------------------------------------------------
    def reset(self) -> None:
        self._store = {s: {} for s in self._scopes}
        self._scope: str | None = None
        self._mode: str | None = None
        self._chunk_id: int | None = None
        self._selected_chunk_ids: list[int] | None = None
        self._pre_rope = False
        # Bumped on every mutation so memoised assemblies are invalidated automatically.
        self._generation = 0
        self._assembly_version: tuple | None = None
        self._assembly_freqs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._assembly_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    def configure(
        self,
        *,
        scope: str | None,
        mode: str | None,
        chunk_id: int | None = None,
        selected_chunk_ids: list[int] | None = None,
        pre_rope: bool = False,
    ) -> None:
        if mode is not None and mode not in _CACHE_MODES:
            raise ValueError(f"Unknown cache mode {mode!r}; expected one of {sorted(_CACHE_MODES)}.")
        self._scope = scope
        self._mode = mode
        self._chunk_id = chunk_id
        self._selected_chunk_ids = list(selected_chunk_ids) if selected_chunk_ids is not None else None
        self._pre_rope = bool(pre_rope)

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def reads(self) -> bool:
        return self._mode in {MODE_REUSE, MODE_REUSE_STORE}

    @property
    def writes(self) -> bool:
        return self._mode in {MODE_STORE, MODE_REUSE_STORE}

    @property
    def generation(self) -> int:
        return self._generation

    def resident_chunk_ids(self, scope: str = "cond") -> set[int]:
        """Which slots currently hold data -- the quantity the window bound is about."""
        return set(self._store.get(scope, {}))

    # -- per-layer hooks used by the DiT blocks --------------------------------------------------
    def _scope_store(self) -> dict[int, dict[int, KVEntry]] | None:
        if self._scope is None:
            return None
        return self._store.setdefault(self._scope, {})

    def read(self, layer_idx: int | None) -> list[KVEntry]:
        """Entries for this layer, in window order. Missing slots are skipped, not errors."""
        scope_store = self._scope_store()
        if scope_store is None or layer_idx is None:
            return []
        entries = []
        for chunk_id in self._selected_chunk_ids or []:
            chunk_store = scope_store.get(chunk_id)
            if chunk_store is None:
                continue
            entry = chunk_store.get(layer_idx)
            if entry is not None:
                entries.append(entry)
        return entries

    def write(self, layer_idx: int | None, key: torch.Tensor, value: torch.Tensor) -> None:
        scope_store = self._scope_store()
        if scope_store is None or layer_idx is None or self._chunk_id is None:
            return
        # `.detach().clone()` rather than storing the tensors as handed over. Both are load-bearing:
        # detach drops the autograd graph these were produced under, and clone gives the cache its own
        # storage. `key`/`value` are slices of the block's fused QKV projection, so keeping them alive
        # as views would pin the whole per-chunk activation -- a leak that grows with clip length and
        # only shows up as an OOM somewhere around chunk 30, far from here.
        scope_store.setdefault(self._chunk_id, {})[layer_idx] = {
            "key": key.detach().clone(),
            "value": value.detach().clone(),
            "pre_rope": self._pre_rope,
        }
        self._generation += 1

    def assemble(
        self,
        layer_idx: int | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cached_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
        memoize: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """:func:`concat_kv_entries` for one layer, memoised across the denoise steps of a chunk.

        Both steps of a chunk see the same window and the same positions, so the concatenated and
        rotated keys are identical between them; recomputing would double the cache work per chunk.

        ``memoize=False`` bypasses the store entirely. Callers must pass it whenever
        :meth:`memo_store` has *not* been primed for the current window and positions -- the per-layer
        entries are keyed by layer alone, so serving them across a window change would return keys
        rotated for the wrong positions.
        """
        if memoize and layer_idx in self._assembly_cache:
            return self._assembly_cache[layer_idx]
        result = concat_kv_entries(self.read(layer_idx), device=device, dtype=dtype, cached_freqs_cis=cached_freqs_cis)
        if memoize:
            self._assembly_cache[layer_idx] = result
        return result

    def _assembly_stamp(self, cached_temporal_ids: torch.Tensor) -> tuple:
        """Identity of the current assembly: scope, window, positions, and mutation counter."""
        ids = torch.as_tensor(cached_temporal_ids, dtype=torch.long)
        return (
            self._scope,
            tuple(self._selected_chunk_ids or ()),
            self._pre_rope,
            (tuple(ids.shape), tuple(ids.reshape(-1).tolist())),
            self._generation,
        )

    def memo_lookup(self, cached_temporal_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Cached rotation tables for this exact window+positions, or ``None``.

        Takes the positions rather than a caller-held stamp on purpose: the stamp includes the
        mutation counter, so a caller that computed it once and reused it would keep hitting a memo
        the cache has since invalidated.
        """
        return self._assembly_freqs if self._assembly_stamp(cached_temporal_ids) == self._assembly_version else None

    def memo_store(
        self, cached_temporal_ids: torch.Tensor, cached_freqs_cis: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self._assembly_version = self._assembly_stamp(cached_temporal_ids)
        self._assembly_freqs = cached_freqs_cis
        self._assembly_cache = {}

    # -- eviction -------------------------------------------------------------------------------
    # Eviction happens *twice* per chunk, around the clean-latent store, with different keep-sets.
    # They are separate named methods because skipping either one is silent: the output stays
    # correct while resident chunks grow without bound until the process runs out of memory.
    def evict(self, chunk_ids_to_keep: set[int]) -> None:
        """Drop every slot outside ``chunk_ids_to_keep``, across all scopes."""
        evicted = False
        for scope_store in self._store.values():
            for chunk_id in [cid for cid in scope_store if cid not in chunk_ids_to_keep]:
                del scope_store[chunk_id]
                evicted = True
        if evicted:
            self._generation += 1
            self._assembly_version = None
            self._assembly_cache = {}

    def evict_before_store(self, history_chunk_ids: Iterable[int], *, has_ref_image: bool) -> None:
        """Keep only the history this chunk attended to; the active chunk is about to be written."""
        keep = {kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if has_ref_image:
            keep.add(kv_cache_memory_id("ref_image"))
        self.evict(keep)

    def evict_after_store(
        self,
        chunk_idx: int,
        *,
        chunk_size: int,
        window_size: int,
        global_sink_chunk: bool,
        has_ref_image: bool,
    ) -> None:
        """Keep only what the *next* chunk will attend to."""
        keep = {
            kv_cache_memory_id("clean", cid)
            for cid in next_selected_chunk_ids(chunk_idx, chunk_size, window_size, global_sink_chunk)
        }
        if has_ref_image:
            keep.add(kv_cache_memory_id("ref_image"))
        self.evict(keep)
