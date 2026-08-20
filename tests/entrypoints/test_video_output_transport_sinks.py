# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Artifact sinks and container formats for the video output transport."""

import io
import tempfile
from types import SimpleNamespace

import av
import numpy as np
import pytest

from vllm_omni.config.server_settings import FileBackend
from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.diffusion.utils.media_utils import (
    default_audio_codec_for_format,
    default_video_codec_for_format,
    media_type_for_format,
)
from vllm_omni.entrypoints.openai.storage import (
    FileStorageHandle,
    LocalStorageManager,
    SaveContext,
    StorageBaseManager,
    UrlStorageHandle,
    get_storage_manager,
    register_storage_backend,
)
from vllm_omni.entrypoints.openai.video_api_utils import (
    _encode_video_bytes,
    resolve_video_encoder_settings,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frames(count: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((count, 32, 48, 3)) * 255).astype(np.uint8)


def _client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(od_config=SimpleNamespace(video_output_transport=VideoOutputTransportConfig(**kwargs)))


# --- container format ------------------------------------------------------


@pytest.mark.parametrize(
    ("output_format", "video_codec", "audio_codec", "media_type"),
    [("mp4", "h264", "aac", "video/mp4"), ("webm", "vp9", "opus", "video/webm")],
)
def test_output_format_selects_a_container_its_codecs_fit(output_format, video_codec, audio_codec, media_type):
    """A container only accepts codecs it can hold, so the format drives both."""
    audio = (np.random.default_rng(1).random(8000) * 0.1).astype(np.float32)
    data = _encode_video_bytes(
        _frames(),
        fps=8,
        audio=audio,
        audio_sample_rate=16000,
        output_format=output_format,
    )
    with av.open(io.BytesIO(data)) as container:
        assert container.streams.video[0].codec_context.name == video_codec
        assert container.streams.audio[0].codec_context.name == audio_codec
    assert media_type_for_format(output_format) == media_type


def test_unknown_output_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported video output format"):
        default_video_codec_for_format("avi")


def test_config_rejects_unknown_format_and_mode():
    with pytest.raises(ValueError, match="output_format must be one of"):
        VideoOutputTransportConfig(output_format="avi")
    with pytest.raises(ValueError, match="transport_mode must be one of"):
        VideoOutputTransportConfig(transport_mode="carrier-pigeon")


def test_codec_default_follows_the_format_not_the_other_way_round():
    """Leaving video_codec unset must not pin h264 into a webm container."""
    assert resolve_video_encoder_settings(_client(output_format="webm")).codec == "libvpx-vp9"
    assert resolve_video_encoder_settings(_client()).codec == "h264"
    assert default_audio_codec_for_format("webm") == "libopus"


def test_streaming_pins_mp4_so_the_artifact_format_cannot_break_fragments():
    """Fragmented-MP4 streaming carries its own container; webm must not leak in."""
    settings = resolve_video_encoder_settings(
        _client(output_format="webm"), None, low_latency=True, force_output_format="mp4"
    )
    assert settings.output_format == "mp4"
    assert settings.codec == "h264"


# --- transport mode / artifact sinks --------------------------------------


def test_transport_mode_is_reported_for_the_sink_decision():
    assert resolve_video_encoder_settings(_client()).transport_mode == "base64"
    assert resolve_video_encoder_settings(_client(transport_mode="url")).transport_mode == "url"


@pytest.mark.asyncio
async def test_local_backend_serves_artifacts_itself_when_not_published():
    with tempfile.TemporaryDirectory() as path:
        manager = LocalStorageManager(storage_path=path)
        await manager.save(b"payload", "clip.mp4")
        handle = await manager.open("clip.mp4")
        assert isinstance(handle, FileStorageHandle)
        # No public URL means the caller falls back to this server's route.
        assert manager.public_url("clip.mp4") is None


@pytest.mark.asyncio
async def test_published_storage_returns_a_url_handle_instead_of_bytes():
    with tempfile.TemporaryDirectory() as path:
        manager = LocalStorageManager(storage_path=path, public_base_url="https://cdn.example.com/v/")
        await manager.save(b"payload", "clip.mp4")
        handle = await manager.open("clip.mp4")
        assert isinstance(handle, UrlStorageHandle)
        assert handle.kind == "url"
        assert handle.url == "https://cdn.example.com/v/clip.mp4"


def test_public_url_rejects_keys_escaping_the_storage_directory():
    with tempfile.TemporaryDirectory() as path:
        manager = LocalStorageManager(storage_path=path, public_base_url="https://cdn.example.com/v")
        with pytest.raises(ValueError, match="Illegal storage key"):
            manager.public_url("../../etc/passwd")


# --- pluggable backends ---------------------------------------------------


class _FakeObjectStore(StorageBaseManager):
    """Stands in for an out-of-tree S3/OSS backend; no client ships in-tree."""

    async def save(self, data, file_name):
        return SaveContext(key=file_name, created_at=0)

    async def delete(self, file_name):
        return True

    async def open(self, storage_key):
        return UrlStorageHandle(url=f"s3://bucket/{storage_key}")

    def public_url(self, storage_key):
        return f"s3://bucket/{storage_key}"


@pytest.mark.asyncio
async def test_an_out_of_tree_backend_can_be_registered_and_returns_url_handles():
    register_storage_backend("test-object-store", lambda config: _FakeObjectStore())
    manager = get_storage_manager(SimpleNamespace(type="test-object-store"))
    assert isinstance(manager, _FakeObjectStore)
    handle = await manager.open("clip.mp4")
    assert isinstance(handle, UrlStorageHandle)
    assert handle.url == "s3://bucket/clip.mp4"


def test_the_builtin_file_backend_still_resolves_through_the_registry():
    with tempfile.TemporaryDirectory() as path:
        manager = get_storage_manager(FileBackend(path=path))
        assert isinstance(manager, LocalStorageManager)
        assert manager.public_base_url is None


def test_file_backend_passes_the_published_base_url_through():
    with tempfile.TemporaryDirectory() as path:
        manager = get_storage_manager(FileBackend(path=path, public_base_url="https://cdn.example.com/v"))
        assert manager.public_url("clip.mp4") == "https://cdn.example.com/v/clip.mp4"


# --- the URL sink used by the synchronous endpoint -------------------------


@pytest.mark.asyncio
async def test_url_sink_falls_back_to_this_server_route_when_unpublished(monkeypatch):
    """Zero-config URL mode: the artifact route serves what storage holds."""
    from vllm_omni.entrypoints.openai import serving_video as serving_video_module
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

    with tempfile.TemporaryDirectory() as path:
        manager = LocalStorageManager(storage_path=path)
        monkeypatch.setattr("vllm_omni.entrypoints.openai.storage.STORAGE_MANAGER", manager)
        serving = object.__new__(OmniOpenAIServingVideo)
        url = await serving_video_module.OmniOpenAIServingVideo._store_video_artifact(serving, b"payload", "mp4")

    assert url.startswith("/v1/videos/artifacts/")
    assert url.endswith(".mp4")


@pytest.mark.asyncio
async def test_url_sink_prefers_the_published_url_when_storage_has_one(monkeypatch):
    from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo

    with tempfile.TemporaryDirectory() as path:
        manager = LocalStorageManager(storage_path=path, public_base_url="https://cdn.example.com/v")
        monkeypatch.setattr("vllm_omni.entrypoints.openai.storage.STORAGE_MANAGER", manager)
        serving = object.__new__(OmniOpenAIServingVideo)
        url = await OmniOpenAIServingVideo._store_video_artifact(serving, b"payload", "webm")

    assert url.startswith("https://cdn.example.com/v/")
    assert url.endswith(".webm")
