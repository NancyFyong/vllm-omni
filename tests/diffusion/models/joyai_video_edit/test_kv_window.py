# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the JoyAI-Video-Edit bounded KV window.

The window is pure bookkeeping, so every failure mode here is silent at runtime: the model still
produces correctly-shaped, finite, plausible-looking video while either drifting or leaking. These
tests are cheap (milliseconds, no weights) and are the only place several of these are catchable.

The contract is taken from upstream ``JoyOmniV2VStreamingSession._denoise_chunk``:

- positions come from ``_gather_window_temporal_ids(gather_chunk_ids, ...)`` then split
  ``cached = window_ids[:-chunk_size]`` / ``current = window_ids[-chunk_size:]``;
- the read set is ``history_chunk_ids = selected_chunk_ids[:-1]`` plus an *appended* reference slot;
- ``evict_kv_cache_chunks`` is called **twice** per chunk, with different keep-sets, around the
  clean-latent store forward.
"""

import pytest
import torch

from vllm_omni.diffusion.models.joyai_video_edit import joyai_video_edit_kv as kv
from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_config import (
    CHUNK_SIZE,
    GLOBAL_SINK_CHUNK,
    KV_CACHE_ID_REF_IMAGE,
    LOCAL_WINDOW_SIZE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

CPU = torch.device("cpu")
GEOMETRY = {
    "chunk_size": CHUNK_SIZE,
    "window_size": LOCAL_WINDOW_SIZE,
    "global_sink_chunk": GLOBAL_SINK_CHUNK,
}


def _windows(total_latent_frames: int) -> list[dict]:
    return kv.get_chunk_windows(total_latent_frames=total_latent_frames, **GEOMETRY)


def _positions(selected_chunk_ids: list[int], total_latent_frames: int) -> torch.Tensor:
    return kv.gather_window_temporal_ids(selected_chunk_ids, CHUNK_SIZE, total_latent_frames, CPU)


# --- window selection ---------------------------------------------------------------------------


def test_window_is_sink_plus_two_most_recent():
    """The exact selection sequence, spelled out rather than recomputed from the same formula."""
    selected = [w["selected_chunk_ids"] for w in _windows(6)]
    assert selected == [[0], [0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]


def test_window_never_exceeds_window_size():
    """The memory bound. ``max(1, ...)`` in the tail arithmetic is easy to port as plain subtraction,
    which grows the window by one from chunk 2 onward."""
    for window in _windows(100):
        assert len(window["selected_chunk_ids"]) <= LOCAL_WINDOW_SIZE


def test_window_always_ends_with_the_active_chunk():
    """``active_chunk_id = selected_chunk_ids[-1]`` upstream, so ordering is load-bearing, not a set."""
    for window in _windows(50):
        assert window["selected_chunk_ids"][-1] == window["chunk_idx"]


def test_window_always_retains_chunk_zero_as_sink():
    for window in _windows(50):
        assert window["selected_chunk_ids"][0] == 0


def test_window_size_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        kv.get_chunk_windows(total_latent_frames=4, chunk_size=1, window_size=0, global_sink_chunk=True)


# --- renumbering --------------------------------------------------------------------------------


def test_renumbered_positions_stay_bounded_over_a_long_rollout():
    """The property that makes the model length-independent.

    Positions are renumbered contiguously from 0 each chunk, so no matter how long the video is the
    DiT never sees a temporal index above ``window_size - 1``. Caching *absolute* positions instead
    would still run, and still look fine early, degrading only as the video gets long.
    """
    total = 100
    for window in _windows(total):
        positions = _positions(window["selected_chunk_ids"], total)
        assert positions.tolist() == list(range(len(window["selected_chunk_ids"])))
        assert int(positions.max()) <= LOCAL_WINDOW_SIZE - 1


def test_renumbering_discards_the_temporal_gap():
    """Chunks 0, 3, 4 become positions 0, 1, 2 -- the gap is deliberately *not* represented.

    This is why keys must be cached pre-RoPE: chunk 3 sits at position 2 while it is the active
    chunk, then at position 1 on the next chunk. Its stored key cannot carry a baked-in rotation.
    """
    assert _positions([0, 3, 4], 100).tolist() == [0, 1, 2]


def test_max_temporal_ids_branch_shifts_instead_of_renumbering():
    """The non-default branch keeps absolute ids, shifted under a ceiling. Upstream leaves
    ``max_temporal_ids`` at ``None``, so this exists to pin the branch, not because it is live."""
    shifted = kv.gather_window_temporal_ids([0, 3, 4], CHUNK_SIZE, 100, CPU, max_temporal_ids=2)
    assert shifted.tolist() == [0, 1, 2]
    unshifted = kv.gather_window_temporal_ids([0, 3, 4], CHUNK_SIZE, 100, CPU, max_temporal_ids=10)
    assert unshifted.tolist() == [0, 3, 4]


# --- the cached/current split -------------------------------------------------------------------


def test_split_gives_cached_exactly_the_history_positions():
    """The alignment contract, and the highest-value assertion in this file.

    ``cached`` must line up one-to-one with the history entries the cache will return. Passing the
    full window table instead (forgetting ``[:-chunk_size]``) shifts every cached key by one position
    -- finite, plausible output whose error grows with chunk index, invisible until a long run.
    """
    total = 20
    for window in _windows(total):
        selected = window["selected_chunk_ids"]
        history = selected[:-1]
        cached, current = kv.split_window_temporal_ids(_positions(selected, total), CHUNK_SIZE)

        assert current.numel() == CHUNK_SIZE
        assert int(current[0]) == len(history)  # active chunk sits just past the history
        if history:
            assert cached is not None
            assert cached.numel() == CHUNK_SIZE * len(history)
            assert cached.tolist() == list(range(len(history)))
        else:
            assert cached is None


def test_split_returns_none_for_the_first_chunk():
    """Chunk 0 has no history; upstream converts the empty tensor to ``None`` so the DiT skips the
    whole cache path rather than concatenating a zero-length key."""
    cached, current = kv.split_window_temporal_ids(_positions([0], 10), CHUNK_SIZE)
    assert cached is None
    assert current.tolist() == [0]


def test_split_rejects_a_table_shorter_than_one_chunk():
    with pytest.raises(ValueError, match="fewer than chunk_size"):
        kv.split_window_temporal_ids(torch.tensor([0]), chunk_size=2)


# --- slot identity ------------------------------------------------------------------------------


def test_reference_slot_cannot_collide_with_any_chunk_id():
    """The reference entry shares the store with clean chunks, so its id must be unreachable by
    ``kv_cache_memory_id("clean", n)``; a 0 sentinel would silently evict the sink chunk."""
    assert kv.kv_cache_memory_id("ref_image") == KV_CACHE_ID_REF_IMAGE
    assert KV_CACHE_ID_REF_IMAGE < 0
    assert all(kv.kv_cache_memory_id("clean", n) != KV_CACHE_ID_REF_IMAGE for n in range(1000))


def test_clean_slot_requires_a_chunk_id():
    with pytest.raises(ValueError, match="required for kind='clean'"):
        kv.kv_cache_memory_id("clean")


def test_reference_slot_is_appended_after_history():
    """Read order is history-then-reference. Prepending it would push every history entry one slot
    down the position table while the table itself stays history-length."""
    assert kv.cache_memory_ids_for_read([0, 3], has_ref_image=True) == [0, 3, KV_CACHE_ID_REF_IMAGE]
    assert kv.cache_memory_ids_for_read([0, 3], has_ref_image=False) == [0, 3]


# --- eviction -----------------------------------------------------------------------------------


def _simulate_rollout(
    total_chunks: int, *, evict_before: bool = True, evict_after: bool = True, has_ref_image: bool = False
) -> list[int]:
    """Run the real per-chunk sequence, returning residency after each chunk.

    Mirrors ``_denoise_chunk``: evict(history) -> store(active) -> evict(next window).
    """
    window = kv.JoyKVWindow(scopes=("cond",))
    if has_ref_image:
        window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=KV_CACHE_ID_REF_IMAGE, pre_rope=False)
        window.write(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))

    residency = []
    for chunk_idx in range(total_chunks):
        selected = _windows(chunk_idx + 1)[-1]["selected_chunk_ids"]
        history = selected[:-1]

        if evict_before:
            window.evict_before_store(history, has_ref_image=has_ref_image)

        # The third forward: write-only, reads nothing (`store_clean_self_only=True` upstream).
        window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=chunk_idx, selected_chunk_ids=[], pre_rope=True)
        window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))

        if evict_after:
            window.evict_after_store(chunk_idx, has_ref_image=has_ref_image, **GEOMETRY)
        residency.append(len(window.resident_chunk_ids("cond")))
    return residency


def test_residency_stays_bounded_over_a_hundred_chunks():
    """Both eviction calls present -- residency plateaus rather than tracking chunk count."""
    residency = _simulate_rollout(100)
    assert max(residency) <= LOCAL_WINDOW_SIZE
    assert residency[-1] == residency[50]  # plateaued, not still creeping


def test_residency_stays_bounded_with_a_reference_image():
    """The reference slot is pinned by both keep-sets, so the bound rises by exactly one, not by N."""
    residency = _simulate_rollout(100, has_ref_image=True)
    assert max(residency) <= LOCAL_WINDOW_SIZE + 1


def test_reference_slot_survives_the_whole_rollout():
    """A keep-set built from chunk ids alone would evict the reference image on chunk 1, silently
    dropping appearance conditioning for every frame after the first."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=KV_CACHE_ID_REF_IMAGE, pre_rope=False)
    window.write(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))

    for chunk_idx in range(20):
        history = _windows(chunk_idx + 1)[-1]["selected_chunk_ids"][:-1]
        window.evict_before_store(history, has_ref_image=True)
        window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=chunk_idx, pre_rope=True)
        window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
        window.evict_after_store(chunk_idx, has_ref_image=True, **GEOMETRY)
        assert KV_CACHE_ID_REF_IMAGE in window.resident_chunk_ids("cond")


def test_dropping_both_evictions_grows_without_bound():
    """The leak guard, and the reason residency is asserted rather than assumed.

    Note what this does *not* claim: either eviction alone already bounds residency (at 3 and 2
    respectively), so this is not a test that both call sites are present -- see
    :func:`test_post_store_eviction_lowers_steady_state_residency` for what the second one buys.
    Without either, the store grows one entry per chunk and a long video OOMs.
    """
    unbounded = _simulate_rollout(40, evict_before=False, evict_after=False)
    assert unbounded == list(range(1, 41))  # one entry per chunk, forever


def test_post_store_eviction_lowers_steady_state_residency():
    """What the second eviction actually buys: the chunk that just left the window is released
    immediately instead of being held until the next chunk's pre-store pass."""
    both = _simulate_rollout(40)
    before_only = _simulate_rollout(40, evict_after=False)
    assert max(both) < max(before_only) <= LOCAL_WINDOW_SIZE


def test_post_store_keep_set_targets_the_next_window():
    """``next_selected_chunk_ids(k)`` must equal the window chunk ``k+1`` will actually request."""
    for chunk_idx in range(20):
        expected = _windows(chunk_idx + 2)[-1]["selected_chunk_ids"]
        assert kv.next_selected_chunk_ids(chunk_idx, **GEOMETRY) == expected


def test_eviction_applies_across_every_scope():
    """Scopes share one keep-set upstream; evicting only the active scope leaks the others."""
    window = kv.JoyKVWindow(scopes=("cond", "uncond"))
    for scope in ("cond", "uncond"):
        for chunk_id in (0, 1, 2):
            window.configure(scope=scope, mode=kv.MODE_STORE, chunk_id=chunk_id)
            window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    window.evict({0, 2})
    assert window.resident_chunk_ids("cond") == {0, 2}
    assert window.resident_chunk_ids("uncond") == {0, 2}


# --- assembly -----------------------------------------------------------------------------------


def _identity_freqs(num_positions: int, head_dim: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position rotation by a distinct angle, so an offset error changes the numbers."""
    angles = torch.arange(1, num_positions + 1, dtype=torch.float32).unsqueeze(-1)
    return angles.cos().expand(num_positions, head_dim).contiguous(), angles.sin().expand(
        num_positions, head_dim
    ).contiguous()


def _entry(value: float, *, pre_rope: bool, seq_len: int = 1, head_dim: int = 4) -> dict:
    tensor = torch.full((1, seq_len, 1, head_dim), value)
    return {"key": tensor.clone(), "value": tensor.clone(), "pre_rope": pre_rope}


def test_assembly_preserves_window_order():
    """Concatenation order must match the position table's order, or keys and positions disagree."""
    entries = [_entry(1.0, pre_rope=False), _entry(2.0, pre_rope=False), _entry(3.0, pre_rope=False)]
    key, value = kv.concat_kv_entries(entries, device=CPU, dtype=torch.float32, cached_freqs_cis=None)
    assert key.shape == (1, 3, 1, 4)
    assert key[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert value[0, :, 0, 0].tolist() == [1.0, 2.0, 3.0]


def test_each_pre_rope_entry_consumes_the_next_position_slice():
    """The pre-RoPE offset must advance by each entry's own length.

    A fixed offset (or one advanced by a constant) rotates later cached entries at the wrong
    position. Verified by rotating the same entries individually at explicit positions.
    """
    freqs = _identity_freqs(3)
    entries = [_entry(1.0, pre_rope=True), _entry(2.0, pre_rope=True, seq_len=2)]
    key, _ = kv.concat_kv_entries(entries, device=CPU, dtype=torch.float32, cached_freqs_cis=freqs)

    from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_rope import apply_rotary_emb

    first = apply_rotary_emb(entries[0]["key"], (freqs[0][0:1], freqs[1][0:1]))
    second = apply_rotary_emb(entries[1]["key"], (freqs[0][1:3], freqs[1][1:3]))
    torch.testing.assert_close(key, torch.cat([first, second], dim=1))


def test_post_rope_entry_is_neither_rotated_nor_consumes_a_position():
    """The reference-image entry is already rotated. Rotating it again is silent corruption of the
    only appearance-conditioning signal, and consuming a slice would shift later entries."""
    freqs = _identity_freqs(3)
    ref, hist = _entry(5.0, pre_rope=False), _entry(1.0, pre_rope=True)
    key, _ = kv.concat_kv_entries([ref, hist], device=CPU, dtype=torch.float32, cached_freqs_cis=freqs)

    torch.testing.assert_close(key[:, 0:1], ref["key"])  # passed through untouched

    from vllm_omni.diffusion.models.joyai_video_edit.joyai_video_edit_rope import apply_rotary_emb

    # The pre-RoPE entry still gets position 0 -- the reference entry did not advance the offset.
    torch.testing.assert_close(key[:, 1:2], apply_rotary_emb(hist["key"], (freqs[0][0:1], freqs[1][0:1])))


def test_empty_window_assembles_to_none():
    """Chunk 0 must yield ``None``, not a zero-length tensor, so the DiT skips the concat entirely."""
    key, value = kv.concat_kv_entries([], device=CPU, dtype=torch.float32, cached_freqs_cis=None)
    assert key is None and value is None


# --- read/write plumbing ------------------------------------------------------------------------


def test_read_returns_entries_in_configured_window_order():
    """Not sorted, not insertion order -- the order ``selected_chunk_ids`` asks for."""
    window = kv.JoyKVWindow(scopes=("cond",))
    for chunk_id in (0, 1, 2):
        window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=chunk_id, pre_rope=True)
        window.write(0, torch.full((1, 1, 1, 4), float(chunk_id)), torch.zeros(1, 1, 1, 4))

    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[2, 0])
    assert [float(e["key"][0, 0, 0, 0]) for e in window.read(0)] == [2.0, 0.0]


def test_read_skips_missing_slots_rather_than_failing():
    """An evicted slot that is still named in the window must be skipped -- upstream relies on this
    during the first chunks, when the window names chunks that were never written."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0, 7])
    assert len(window.read(0)) == 1


def test_store_forward_reads_nothing():
    """With ``store_clean_self_only=True`` (upstream default) the third forward is write-only: its
    selected set is empty, so it attends to its own chunk alone even though history is resident."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))

    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=1, selected_chunk_ids=[], pre_rope=True)
    assert window.read(0) == []
    assert window.writes and not window.reads


def test_scopes_are_isolated():
    """A cond/uncond mix-up is invisible in shapes and would only show as degraded guidance."""
    window = kv.JoyKVWindow(scopes=("cond", "uncond"))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))

    window.configure(scope="uncond", mode=kv.MODE_REUSE, selected_chunk_ids=[0])
    assert window.read(0) == []


def test_write_records_the_pre_rope_flag_it_was_configured_with():
    """The flag decides whether the assembler rotates the entry later; losing it silently
    double-rotates or never-rotates."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=KV_CACHE_ID_REF_IMAGE, pre_rope=False)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))

    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0, KV_CACHE_ID_REF_IMAGE])
    assert [e["pre_rope"] for e in window.read(0)] == [True, False]


def test_write_does_not_alias_the_caller_tensor():
    """Latents are mutated in place across denoise steps; storing a view would corrupt history."""
    window = kv.JoyKVWindow(scopes=("cond",))
    key = torch.zeros(1, 1, 1, 4)
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, key, key)
    key.fill_(9.0)

    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0])
    assert float(window.read(0)[0]["key"].max()) == 0.0


def test_layers_are_stored_independently():
    """40 blocks share one store keyed by layer; collapsing the key makes every block read block 39."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    for layer_idx in range(3):
        window.write(layer_idx, torch.full((1, 1, 1, 4), float(layer_idx)), torch.zeros(1, 1, 1, 4))

    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0])
    assert [float(window.read(i)[0]["key"][0, 0, 0, 0]) for i in range(3)] == [0.0, 1.0, 2.0]


def test_unknown_cache_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown cache mode"):
        kv.JoyKVWindow().configure(scope="cond", mode="write")


# --- assembly memoisation -----------------------------------------------------------------------


def test_memo_is_invalidated_by_a_write():
    """Both denoise steps of a chunk reuse the assembly; the store forward then mutates the cache.
    A memo that survives it serves the previous chunk's keys forever."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_REUSE_STORE, chunk_id=0, selected_chunk_ids=[0], pre_rope=True)
    positions = torch.tensor([0])
    window.memo_store(positions, _identity_freqs(1))
    assert window.memo_lookup(positions) is not None

    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    assert window.memo_lookup(positions) is None


def test_memo_is_invalidated_by_changed_positions():
    """Same window, renumbered positions -- the cached keys must be re-rotated. This is the memo's
    subtlest failure: identical chunk ids across chunks do *not* imply identical positions."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0, 1], pre_rope=True)
    window.memo_store(torch.tensor([0, 1]), _identity_freqs(2))

    assert window.memo_lookup(torch.tensor([0, 1])) is not None
    assert window.memo_lookup(torch.tensor([1, 2])) is None


def test_memo_is_invalidated_by_a_changed_window():
    """Same positions, different chunk ids -- e.g. window [0,1,2] becoming [0,2,3] with identical
    renumbered positions [0,1,2]. Keying on positions alone would serve the wrong chunks' keys."""
    window = kv.JoyKVWindow(scopes=("cond",))
    positions = torch.tensor([0, 1])
    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0, 1], pre_rope=True)
    window.memo_store(positions, _identity_freqs(2))

    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0, 2], pre_rope=True)
    assert window.memo_lookup(positions) is None


def test_memo_is_invalidated_by_eviction():
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=3, pre_rope=True)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0], pre_rope=True)
    positions = torch.tensor([0])
    window.memo_store(positions, _identity_freqs(1))
    assert window.memo_lookup(positions) is not None

    window.evict({0})
    assert window.memo_lookup(positions) is None


def test_memo_store_clears_the_per_layer_assembly_cache():
    """The per-layer assemblies are only valid for the rotation tables that built them, so replacing
    the tables must drop them. A stale layer cache silently reuses the previous chunk's rotation."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.full((1, 1, 1, 4), 2.0), torch.zeros(1, 1, 1, 4))
    window.configure(scope="cond", mode=kv.MODE_REUSE, selected_chunk_ids=[0], pre_rope=True)

    no_rotation = (torch.ones(1, 4), torch.zeros(1, 4))
    quarter_turn = (torch.zeros(1, 4), torch.ones(1, 4))

    unrotated = window.assemble(0, device=CPU, dtype=torch.float32, cached_freqs_cis=no_rotation)[0]
    # Same layer again with different tables: served from the layer cache, so unchanged...
    assert torch.equal(window.assemble(0, device=CPU, dtype=torch.float32, cached_freqs_cis=quarter_turn)[0], unrotated)
    # ...until new tables are published, which invalidates it.
    window.memo_store(torch.tensor([0]), quarter_turn)
    rotated = window.assemble(0, device=CPU, dtype=torch.float32, cached_freqs_cis=quarter_turn)[0]
    assert not torch.equal(rotated, unrotated)


def test_reset_clears_every_slot():
    """One window instance serves consecutive requests; leftover state would bleed across videos."""
    window = kv.JoyKVWindow(scopes=("cond",))
    window.configure(scope="cond", mode=kv.MODE_STORE, chunk_id=0, pre_rope=True)
    window.write(0, torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    window.reset()
    assert window.resident_chunk_ids("cond") == set()
    assert window.mode is None
