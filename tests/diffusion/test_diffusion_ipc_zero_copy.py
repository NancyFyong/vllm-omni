# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Zero-copy (borrowed) shared-memory transport for diffusion outputs."""

import contextlib
from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput, VideoOutputTransportConfig
from vllm_omni.diffusion.ipc import (
    _SHM_TENSOR_THRESHOLD,
    _pack_value_if_large,
    _unpack_if_shm_handle,
    borrowed_diffusion_output,
    pack_diffusion_output_shm,
    release_borrowed_segments,
    unpack_diffusion_output_shm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_SMALL_THRESHOLD = 1024


def _segment_exists(name: str) -> bool:
    try:
        existing = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    existing.close()
    return True


def _large_tensor(numel: int = 4096) -> torch.Tensor:
    return torch.arange(numel, dtype=torch.float32)


def test_borrowed_tensor_aliases_the_shared_segment_instead_of_copying() -> None:
    """The borrowed tensor must alias the segment, not copy it.

    Writing through an independent handle to the same segment must be visible
    through the borrowed tensor. If a future change reintroduces a defensive
    ``.copy()`` on the borrow path, the tensor stops observing the write and
    this test fails.
    """
    tensor = _large_tensor()
    handle = _pack_value_if_large(tensor, threshold=_SMALL_THRESHOLD)
    assert isinstance(handle, dict) and handle["__tensor_shm__"] is True

    borrowed: list[str] = []
    borrowed_tensor = _unpack_if_shm_handle(handle, borrow=True, borrowed=borrowed)
    try:
        assert isinstance(borrowed_tensor, torch.Tensor)
        torch.testing.assert_close(borrowed_tensor, tensor)

        writer = shared_memory.SharedMemory(name=handle["name"])
        try:
            view = np.ndarray(
                handle["shape"],
                dtype=np.dtype(handle["numpy_dtype"]),
                buffer=writer.buf[: handle["nbytes"]],
            )
            view[0] = 4242.0
        finally:
            writer.close()

        # Shared storage, not a private copy.
        assert borrowed_tensor[0].item() == 4242.0
    finally:
        del borrowed_tensor
        release_borrowed_segments(borrowed)


def test_borrow_keeps_segment_alive_until_released() -> None:
    """Borrowing must not unlink on read, and releasing must not leak.

    The copy path unlinks during the read, so ownership is trivial. Borrowing
    moves ownership to the reader; if release ever stops unlinking, /dev/shm
    fills up with multi-GB video segments.
    """
    handle = _pack_value_if_large(_large_tensor(), threshold=_SMALL_THRESHOLD)
    name = handle["name"]

    borrowed: list[str] = []
    borrowed_tensor = _unpack_if_shm_handle(handle, borrow=True, borrowed=borrowed)
    assert borrowed == [name]
    assert _segment_exists(name), "borrow must not unlink the segment on read"

    # Release while the borrowed tensor is still alive: unlink must still win
    # (close() raises BufferError on exported buffers and is suppressed).
    del borrowed_tensor
    release_borrowed_segments(borrowed)
    assert not _segment_exists(name), "released segment leaked into /dev/shm"


def test_release_unlinks_segment_despite_a_live_borrow_reference() -> None:
    """Regression guard for the BufferError ownership trap.

    ``SharedMemory.close()`` raises BufferError while a borrowed tensor still
    exports the mmap. Closing before unlinking would propagate that error and
    skip the unlink, leaking a multi-GB segment into /dev/shm. Unlinking first
    makes release unconditional.

    The borrowed tensor is deliberately still referenced here, but must NOT be
    touched after release: that is a documented use-after-free (release() frees
    the parent memoryview the borrow was sliced from).
    """
    handle = _pack_value_if_large(_large_tensor(), threshold=_SMALL_THRESHOLD)
    name = handle["name"]

    borrowed: list[str] = []
    borrowed_tensor = _unpack_if_shm_handle(handle, borrow=True, borrowed=borrowed)
    assert isinstance(borrowed_tensor, torch.Tensor)

    release_borrowed_segments(borrowed)

    assert not _segment_exists(name)
    # Releasing twice must be a harmless no-op (idempotent cleanup paths).
    release_borrowed_segments(borrowed)
    assert not _segment_exists(name)


def test_copy_mode_is_unchanged_and_consumes_the_segment() -> None:
    """Default transport must keep the historical copy-and-unlink semantics."""
    tensor = _large_tensor()
    handle = _pack_value_if_large(tensor, threshold=_SMALL_THRESHOLD)
    name = handle["name"]

    unpacked = _unpack_if_shm_handle(handle)

    assert isinstance(unpacked, torch.Tensor)
    torch.testing.assert_close(unpacked, tensor)
    assert not _segment_exists(name), "copy mode must unlink on read"


def test_borrowed_diffusion_output_releases_segments_on_exit() -> None:
    output = DiffusionOutput(output={"video": _large_tensor()})
    pack_diffusion_output_shm(output, threshold=_SMALL_THRESHOLD)
    name = output.output["video"]["name"]

    with borrowed_diffusion_output(output) as borrowed_output:
        assert isinstance(borrowed_output.output["video"], torch.Tensor)
        assert _segment_exists(name)
        # Copy out before the block ends: the borrow dies with the block.
        survivor = borrowed_output.output["video"].clone()

    assert not _segment_exists(name)
    assert survivor[0].item() == 0.0


def test_borrowed_diffusion_output_releases_segments_on_exception() -> None:
    """A consumer crash must not leak a multi-GB video segment."""
    output = DiffusionOutput(output={"video": _large_tensor()})
    pack_diffusion_output_shm(output, threshold=_SMALL_THRESHOLD)
    name = output.output["video"]["name"]

    with contextlib.suppress(RuntimeError):
        with borrowed_diffusion_output(output):
            assert _segment_exists(name)
            raise RuntimeError("consumer blew up")

    assert not _segment_exists(name)


def test_borrowed_diffusion_output_with_borrow_disabled_falls_back_to_copying() -> None:
    """One call site serves both transport modes."""
    output = DiffusionOutput(output={"video": _large_tensor()})
    pack_diffusion_output_shm(output, threshold=_SMALL_THRESHOLD)
    name = output.output["video"]["name"]

    with borrowed_diffusion_output(output, borrow=False) as copied_output:
        video = copied_output.output["video"]
        assert isinstance(video, torch.Tensor)
        # Copy mode already unlinked the segment inside the block.
        assert not _segment_exists(name)

    # Tensor stays valid after the block because it is a private copy.
    assert video[0].item() == 0.0


def test_numpy_payloads_are_borrowed_without_copying() -> None:
    frames = np.arange(4096, dtype=np.float32)
    output = DiffusionOutput(output=frames)
    pack_diffusion_output_shm(output, threshold=_SMALL_THRESHOLD)
    assert output.output["__ndarray_shm__"] is True
    name = output.output["name"]

    with borrowed_diffusion_output(output) as borrowed_output:
        borrowed_array = borrowed_output.output
        assert isinstance(borrowed_array, np.ndarray)
        np.testing.assert_array_equal(borrowed_array, frames)
        # A borrowed NumPy payload must not own its buffer.
        assert borrowed_array.base is not None

    assert not _segment_exists(name)


def test_configured_threshold_packs_tensors_the_default_would_keep_inline() -> None:
    """The threshold is deployment configuration, not a constant."""
    config = VideoOutputTransportConfig(shm_threshold_bytes=_SMALL_THRESHOLD)
    # Comfortably below the module default, comfortably above the configured one.
    tensor = torch.arange(_SMALL_THRESHOLD, dtype=torch.float32)
    assert tensor.nelement() * tensor.element_size() < _SHM_TENSOR_THRESHOLD

    inline = _pack_value_if_large(tensor)
    assert inline is tensor, "default threshold should keep this tensor inline"

    packed = _pack_value_if_large(tensor, threshold=config.shm_threshold_bytes)
    try:
        assert isinstance(packed, dict) and packed["__tensor_shm__"] is True
    finally:
        with contextlib.suppress(FileNotFoundError):
            _unpack_if_shm_handle(packed)


def test_unpack_without_borrow_list_still_releases_by_default() -> None:
    """Callers that ignore the borrowed list must not silently leak."""
    output = DiffusionOutput(output={"video": _large_tensor()})
    pack_diffusion_output_shm(output, threshold=_SMALL_THRESHOLD)
    name = output.output["video"]["name"]

    unpack_diffusion_output_shm(output)

    assert isinstance(output.output["video"], torch.Tensor)
    assert not _segment_exists(name)


@pytest.mark.parametrize(
    "kwargs",
    [{"shm_threshold_bytes": 0}, {"shm_threshold_bytes": -1}, {"video_codec_options": "ultrafast"}],
)
def test_invalid_transport_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        VideoOutputTransportConfig(**kwargs)


def test_transport_config_defaults_preserve_current_behaviour() -> None:
    config = VideoOutputTransportConfig()
    assert config.shm_threshold_bytes == _SHM_TENSOR_THRESHOLD
    assert config.enable_device_postprocess is False
