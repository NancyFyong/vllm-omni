# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""IPC utilities for transferring large tensors via POSIX shared memory.

Used by Hop1 (GPU worker <-> scheduler) to avoid pickling large video tensors
through the MessageQueue. Tensors above the packing threshold are copied
into a named shared-memory segment; only a lightweight metadata dict is
serialised through the queue.

Two read strategies exist, selected by
:class:`~vllm_omni.diffusion.data.VideoOutputTransportConfig`:

``copy`` (default)
    The reader copies the payload out of the segment and unlinks it. Ownership
    is trivially correct because the segment dies on read, at the cost of one
    extra full host copy per hop.

``shared_memory`` (zero-copy, opt-in)
    The reader maps the segment and wraps it *without* copying, for co-located
    consumers (e.g. verl-omni rollout workers on the same host). This removes
    the read-side copy but transfers ownership to the reader, which must
    release the segment -- always go through :func:`borrowed_diffusion_output`.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionOutput

_SHM_TENSOR_THRESHOLD = 1_000_000  # 1 MB
DIFFUSION_RPC_RESULT_ENVELOPE = "diffusion_rpc_result"

# Segments currently borrowed zero-copy by this process, keyed by segment name.
# Borrowed arrays point straight at ``shm.buf``, so the SharedMemory object has
# to outlive them: dropping the last reference unmaps the buffer and would
# leave dangling tensors. Entries are removed by ``release_borrowed_segments``.
_BORROWED_SEGMENTS: dict[str, Any] = {}
_BORROWED_SEGMENTS_LOCK = threading.Lock()


def _array_to_shm(array: np.ndarray) -> dict[str, Any]:
    """Copy a contiguous NumPy-compatible array into shared memory."""
    from multiprocessing import shared_memory

    if array.dtype.hasobject:
        raise TypeError("NumPy object arrays cannot be transferred through raw shared memory")

    array = np.ascontiguousarray(array)
    nbytes = array.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf[:nbytes])
    np.copyto(shm_array, array)
    handle = {
        "name": shm.name,
        "shape": list(array.shape),
        "numpy_dtype": str(array.dtype),
        "nbytes": nbytes,
    }
    shm.close()
    return handle


def _array_from_shm(
    handle: dict[str, Any],
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> np.ndarray:
    """Read an array from shared memory.

    With ``borrow=False`` (default) the array is copied out and the segment is
    closed and unlinked, so nothing survives the call.

    With ``borrow=True`` the returned array is a **view onto the shared
    segment** -- no copy is made and the segment is kept alive in
    ``_BORROWED_SEGMENTS``. The segment name is appended to *borrowed* so the
    caller can release it later. Because the memory is shared, mutating the
    returned array is visible to every other holder of the same segment, and
    the array is only valid until ``release_borrowed_segments`` is called for
    that name.
    """
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=handle["name"])
    if borrow:
        array = np.ndarray(
            handle["shape"],
            dtype=np.dtype(handle["numpy_dtype"]),
            buffer=shm.buf[: handle["nbytes"]],
        )
        with _BORROWED_SEGMENTS_LOCK:
            # A duplicate name means the same segment was borrowed twice; keep
            # the first SharedMemory object so the existing view stays mapped.
            _BORROWED_SEGMENTS.setdefault(handle["name"], shm)
        if borrowed is not None:
            borrowed.append(handle["name"])
        return array

    try:
        array = np.ndarray(
            handle["shape"],
            dtype=np.dtype(handle["numpy_dtype"]),
            buffer=shm.buf[: handle["nbytes"]],
        ).copy()
    finally:
        shm.close()
        shm.unlink()
    return array


def release_borrowed_segments(names: list[str] | None) -> None:
    """Release segments handed out by ``_array_from_shm(borrow=True)``.

    Unlinks before closing: removing the name always succeeds even while a
    borrowed tensor still exports the mmap, so a segment can never be leaked
    into ``/dev/shm`` no matter what the consumer holds.

    .. warning::
        This invalidates every tensor borrowed from these segments, exactly like
        ``free()``. ``close()`` releases the parent memoryview the borrowed
        arrays were sliced from, so touching a borrowed tensor afterwards is a
        use-after-free and will segfault. Consume borrowed tensors inside
        :func:`borrowed_diffusion_output` and copy out whatever must outlive it.
    """
    if not names:
        return
    for name in names:
        with _BORROWED_SEGMENTS_LOCK:
            shm = _BORROWED_SEGMENTS.pop(name, None)
        if shm is None:
            continue
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()
        with contextlib.suppress(BufferError):
            shm.close()


def _tensor_to_shm(
    tensor: torch.Tensor,
    d2h_stream: torch.Stream | None = None,
) -> dict[str, Any]:
    """Copy a tensor into POSIX shared memory and return a metadata handle.

    The shared memory segment remains alive after this call (the local fd is
    closed, but the segment persists until ``_tensor_from_shm`` unlinks it).

    If *d2h_stream* is provided, the D2H copy uses ``copy_()`` with
    ``pin_memory=True`` on that stream instead of the synchronous ``.cpu()``
    path.  The caller must synchronize *d2h_stream* after all tensors are
    packed.
    """
    original_dtype = tensor.dtype
    if d2h_stream is not None:
        # Non-blocking D2H: copy on side stream to pinned CPU memory.
        old_stream = torch.accelerator.current_stream()
        torch.accelerator.set_stream(d2h_stream)
        try:
            t = tensor.detach()
            if original_dtype == torch.bfloat16:
                t = t.to(torch.float32)
            cpu = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            cpu.copy_(t, non_blocking=True)
        finally:
            torch.accelerator.set_stream(old_stream)
        d2h_stream.synchronize()
        tensor = cpu
    else:
        tensor = tensor.detach().cpu().contiguous()
        if original_dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
    handle = _array_to_shm(tensor.numpy())
    handle.update(
        {
            "__tensor_shm__": True,
            "torch_dtype": str(original_dtype),
        }
    )
    return handle


def _tensor_from_shm(
    handle: dict[str, Any],
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> torch.Tensor:
    """Reconstruct a tensor from a shared-memory handle.

    ``borrow=False`` copies and frees the segment. ``borrow=True`` wraps the
    segment without copying (see ``_array_from_shm``).

    Note: a tensor packed from bfloat16 was widened to float32 for transport,
    so restoring its dtype necessarily allocates. Such tensors still cost one
    copy when borrowed; uint8/float32 video payloads borrow with no copy.
    """
    tensor = torch.from_numpy(_array_from_shm(handle, borrow=borrow, borrowed=borrowed))
    # Restore the original dtype if it differs from the numpy-compatible
    # dtype used for the SHM transfer (e.g. bfloat16 → float32 → bfloat16).
    torch_dtype_str = handle.get("torch_dtype", "")
    if torch_dtype_str:
        original_dtype = getattr(torch, torch_dtype_str.replace("torch.", ""), None)
        if original_dtype is not None and tensor.dtype != original_dtype:
            tensor = tensor.to(original_dtype)
    return tensor


def _pack_tensor_if_large(
    val: torch.Tensor,
    d2h_stream: torch.Stream | None = None,
    threshold: int | None = None,
) -> torch.Tensor | dict:
    """Replace a tensor with an SHM handle if it exceeds the threshold.

    Batch outputs are split into per-request views that share one storage.
    Pickle serializes a view's whole storage, so a batch of small views costs
    the full batch tensor once per request. Size the decision on the storage
    to keep those views off the wire; packing copies just the view.
    """
    if threshold is None:
        threshold = _SHM_TENSOR_THRESHOLD
    view_bytes = val.nelement() * val.element_size()
    try:
        storage_bytes = val.untyped_storage().nbytes()
    except Exception:
        storage_bytes = view_bytes
    if max(view_bytes, storage_bytes) > threshold:
        return _tensor_to_shm(val, d2h_stream=d2h_stream)
    return val


def _ndarray_to_shm(array: np.ndarray) -> dict[str, Any]:
    """Copy a contiguous NumPy array into POSIX shared memory."""
    handle = _array_to_shm(array)
    handle["__ndarray_shm__"] = True
    return handle


def _ndarray_from_shm(
    handle: dict[str, Any],
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> np.ndarray:
    """Reconstruct a NumPy array from shared memory.

    Frees the segment unless *borrow* is set (see ``_array_from_shm``).
    """
    return _array_from_shm(handle, borrow=borrow, borrowed=borrowed)


def _pack_value_if_large(
    val: object,
    d2h_stream: torch.Stream | None = None,
    threshold: int | None = None,
) -> object:
    """Recursively replace large tensors with SHM handles.

    Walks the container shapes pipelines return as ``DiffusionOutput.output``:
    bare tensors, dicts (e.g. Cosmos3 ``{"image"/"video": ...}``), and
    tuples/lists (e.g. LTX2 and DreamID ``(video, audio)``). Other values pass
    through unchanged. ``_unpack_if_shm_handle`` must mirror these shapes — keep
    the two in sync.
    """
    if threshold is None:
        threshold = _SHM_TENSOR_THRESHOLD
    if isinstance(val, torch.Tensor):
        return _pack_tensor_if_large(val, d2h_stream=d2h_stream, threshold=threshold)
    if isinstance(val, np.ndarray):
        return _ndarray_to_shm(val) if not val.dtype.hasobject and val.nbytes > threshold else val
    if isinstance(val, dict):
        return {
            key: _pack_value_if_large(value, d2h_stream=d2h_stream, threshold=threshold) for key, value in val.items()
        }
    if isinstance(val, list):
        return [_pack_value_if_large(item, d2h_stream=d2h_stream, threshold=threshold) for item in val]
    if isinstance(val, tuple):
        return tuple(_pack_value_if_large(item, d2h_stream=d2h_stream, threshold=threshold) for item in val)
    return val


def _unpack_if_shm_handle(
    val: object,
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> object:
    """Reconstruct tensors from SHM handles, mirroring ``_pack_value_if_large``."""
    if isinstance(val, dict) and val.get("__tensor_shm__"):
        return _tensor_from_shm(val, borrow=borrow, borrowed=borrowed)
    if isinstance(val, dict) and val.get("__ndarray_shm__"):
        return _ndarray_from_shm(val, borrow=borrow, borrowed=borrowed)
    if isinstance(val, dict):
        return {key: _unpack_if_shm_handle(value, borrow=borrow, borrowed=borrowed) for key, value in val.items()}
    if isinstance(val, list):
        return [_unpack_if_shm_handle(item, borrow=borrow, borrowed=borrowed) for item in val]
    if isinstance(val, tuple):
        return tuple(_unpack_if_shm_handle(item, borrow=borrow, borrowed=borrowed) for item in val)
    return val


def _pack_diffusion_fields(
    output: DiffusionOutput,
    d2h_stream: torch.Stream | None = None,
    threshold: int | None = None,
) -> DiffusionOutput:
    if output.output is not None:
        output.output = _pack_value_if_large(output.output, d2h_stream=d2h_stream, threshold=threshold)
    if output.trajectory_latents is not None and isinstance(output.trajectory_latents, torch.Tensor):
        output.trajectory_latents = _pack_tensor_if_large(
            output.trajectory_latents, d2h_stream=d2h_stream, threshold=threshold
        )
    if output.trajectory_timesteps is not None and isinstance(output.trajectory_timesteps, torch.Tensor):
        output.trajectory_timesteps = _pack_tensor_if_large(
            output.trajectory_timesteps, d2h_stream=d2h_stream, threshold=threshold
        )
    if output.trajectory_log_probs is not None and isinstance(output.trajectory_log_probs, torch.Tensor):
        output.trajectory_log_probs = _pack_tensor_if_large(
            output.trajectory_log_probs, d2h_stream=d2h_stream, threshold=threshold
        )
    return output


def _is_rpc_result_envelope(output: object) -> bool:
    return isinstance(output, dict) and output.get("type") == DIFFUSION_RPC_RESULT_ENVELOPE


def pack_diffusion_output_shm(
    output: object,
    d2h_stream: torch.Stream | None = None,
    threshold: int | None = None,
) -> object:
    """Replace large tensors in diffusion worker outputs with SHM handles.

    Supports a bare ``DiffusionOutput``, a wrapper object carrying one in
    ``.result`` (for example ``RunnerOutput``), an RPC result envelope carrying
    the diffusion output in ``["result"]``, a batch wrapper carrying
    ``RunnerOutput`` objects in ``.runner_outputs``, or a DP-tagged dict
    ``{"dp_rank": int, "output": DiffusionOutput}`` used by DP multi-concurrency.

    If *d2h_stream* is provided, D2H copies use that stream (non-blocking on
    the default stream).  The caller must synchronize *d2h_stream* afterward.

    *threshold* overrides the byte size above which a tensor moves through
    shared memory; it defaults to ``_SHM_TENSOR_THRESHOLD``. Callers holding an
    ``OmniDiffusionConfig`` should pass
    ``config.video_output_transport.shm_threshold_bytes``.
    """
    if isinstance(output, DiffusionOutput):
        return _pack_diffusion_fields(output, d2h_stream=d2h_stream, threshold=threshold)

    # DP multi-concurrency: {"dp_rank": int, "output": DiffusionOutput}
    if isinstance(output, dict) and "dp_rank" in output and "output" in output:
        inner = output["output"]
        if isinstance(inner, DiffusionOutput):
            output["output"] = _pack_diffusion_fields(inner, d2h_stream=d2h_stream, threshold=threshold)
        return output

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        output["result"] = pack_diffusion_output_shm(result, d2h_stream=d2h_stream, threshold=threshold)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _pack_diffusion_fields(result, d2h_stream=d2h_stream, threshold=threshold)

    runner_outputs = getattr(output, "runner_outputs", None)
    if isinstance(runner_outputs, list):
        for runner_output in runner_outputs:
            pack_diffusion_output_shm(runner_output, d2h_stream=d2h_stream, threshold=threshold)
    return output


def _unpack_diffusion_fields(
    output: DiffusionOutput,
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> DiffusionOutput:
    output.output = _unpack_if_shm_handle(output.output, borrow=borrow, borrowed=borrowed)
    output.trajectory_latents = _unpack_if_shm_handle(output.trajectory_latents, borrow=borrow, borrowed=borrowed)
    output.trajectory_timesteps = _unpack_if_shm_handle(output.trajectory_timesteps, borrow=borrow, borrowed=borrowed)
    output.trajectory_log_probs = _unpack_if_shm_handle(output.trajectory_log_probs, borrow=borrow, borrowed=borrowed)
    return output


def unpack_diffusion_output_shm(
    output: object,
    *,
    borrow: bool = False,
    borrowed: list[str] | None = None,
) -> object:
    """Reconstruct tensors from SHM handles in diffusion worker outputs.

    By default each payload is copied out of shared memory and its segment is
    unlinked. With ``borrow=True`` the tensors alias the shared segments
    instead (no copy) and the touched segment names are collected into
    *borrowed*, which the caller must hand to ``release_borrowed_segments``.
    Prefer :func:`borrowed_diffusion_output`, which does that automatically.
    """
    if isinstance(output, DiffusionOutput):
        return _unpack_diffusion_fields(output, borrow=borrow, borrowed=borrowed)

    # DP multi-concurrency: {"dp_rank": int, "output": DiffusionOutput}
    if isinstance(output, dict) and "dp_rank" in output and "output" in output:
        inner = output["output"]
        if isinstance(inner, DiffusionOutput):
            output["output"] = _unpack_diffusion_fields(inner, borrow=borrow, borrowed=borrowed)
        return output

    if _is_rpc_result_envelope(output):
        result = output.get("result")
        output["result"] = unpack_diffusion_output_shm(result, borrow=borrow, borrowed=borrowed)
        return output

    result = getattr(output, "result", None)
    if isinstance(result, DiffusionOutput):
        output.result = _unpack_diffusion_fields(result, borrow=borrow, borrowed=borrowed)

    runner_outputs = getattr(output, "runner_outputs", None)
    if isinstance(runner_outputs, list):
        for runner_output in runner_outputs:
            unpack_diffusion_output_shm(runner_output, borrow=borrow, borrowed=borrowed)
    return output


@contextlib.contextmanager
def borrowed_diffusion_output(output: object, *, borrow: bool = True):
    """Yield *output* with its SHM payloads materialised, releasing on exit.

    This is the supported entry point for zero-copy consumption. Every borrowed
    segment is released on exit, including on error, so a segment can never
    outlive its consumer and leak ``/dev/shm``.

    .. warning::
        With ``borrow=True`` the yielded tensors alias shared memory and become
        invalid when the block exits -- using them afterwards is a
        use-after-free and will segfault. Copy out (``.clone()``) anything that
        must outlive the block.

    Pass ``borrow=False`` to get the copying behaviour through the same call
    site; the block then yields private copies that stay valid afterwards and
    the release step is a no-op.
    """
    borrowed: list[str] = []
    try:
        yield unpack_diffusion_output_shm(output, borrow=borrow, borrowed=borrowed)
    finally:
        release_borrowed_segments(borrowed)
