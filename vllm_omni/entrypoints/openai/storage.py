import asyncio
import contextlib
import os
import stat
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import Any, Generic, Literal, TypeVar

from vllm.logger import init_logger

from vllm_omni.config.server_settings import SERVER_SETTINGS_CONFIG, FileBackend

logger = init_logger(__name__)


@dataclass
class SaveContext:
    key: str
    created_at: int
    expires_at: int | None = None


@dataclass(frozen=True)
class BaseStorageHandle:
    kind: str


@dataclass(frozen=True)
class FileStorageHandle(BaseStorageHandle):
    path: str
    kind: Literal["path"] = field(default="path", init=False)


@dataclass(frozen=True)
class UrlStorageHandle(BaseStorageHandle):
    """An artifact the client should fetch itself instead of being served inline.

    Returned by backends that offload artifacts somewhere already reachable by
    the client (a shared filesystem behind a static server, or an object store).
    """

    url: str
    kind: Literal["url"] = field(default="url", init=False)


K = TypeVar("K", bound=BaseStorageHandle, covariant=True)


class StorageBaseManager(Generic[K], ABC):
    @abstractmethod
    async def save(self, *args, **kwargs) -> SaveContext:
        pass

    @abstractmethod
    async def delete(self, *args, **kwargs) -> bool:
        pass

    async def start(self, *args, **kwargs) -> None:
        pass

    async def stop(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    async def open(self, storage_key: str) -> K | None:
        pass

    def public_url(self, storage_key: str) -> str | None:
        """A client-reachable URL for the artifact, when the backend has one.

        ``None`` means the artifact is only reachable through this server, so
        callers should fall back to serving it themselves.
        """
        return None


class LocalStorageManager(StorageBaseManager[FileStorageHandle | UrlStorageHandle]):
    def __init__(self, storage_path: str, max_concurrency: int = 4, public_base_url: str | None = None):
        self.storage_path = os.path.realpath(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

        # Set when the storage directory is already published by something else
        # (a static server or CDN in front of a shared filesystem), which lets us
        # hand clients a URL instead of streaming bytes through this process.
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None

        self._io_semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def public_url(self, storage_key: str) -> str | None:
        if self.public_base_url is None:
            return None
        # Validates the key before it is handed out as a URL.
        self.get_full_file_path(storage_key)
        return f"{self.public_base_url}/{storage_key}"

    async def open(self, storage_key: str) -> FileStorageHandle | UrlStorageHandle | None:
        local_file = self.get_full_file_path(storage_key)
        if not os.path.exists(local_file):
            return None
        public_url = self.public_url(storage_key)
        if public_url is not None:
            return UrlStorageHandle(url=public_url)
        return FileStorageHandle(path=local_file)

    def _save_sync(self, data: bytes, file_name: str) -> SaveContext:
        filename = self.get_full_file_path(file_name)
        tmp_name: str | None = None

        try:
            with NamedTemporaryFile("wb", dir=self.storage_path, delete=False) as f:
                tmp_name = f.name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, filename)
            response = SaveContext(key=file_name, created_at=int(time.time()))
            return response
        except Exception:
            if tmp_name is not None:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            raise

    async def save(self, data: bytes, file_name: str) -> SaveContext:
        async with self._io_semaphore:
            return await asyncio.to_thread(self._save_sync, data, file_name)

    def _delete_sync(self, file_name: str) -> bool:
        try:
            os.remove(self.get_full_file_path(file_name))
        except FileNotFoundError:
            return False
        return True

    async def delete(self, file_name: str) -> bool:
        async with self._io_semaphore:
            return await asyncio.to_thread(self._delete_sync, file_name)

    def exists(self, file_name: str) -> bool:
        return os.path.exists(self.get_full_file_path(file_name))

    def get_full_file_path(self, file_name: str) -> str:
        full = os.path.realpath(os.path.join(self.storage_path, file_name))

        if os.path.commonpath([self.storage_path, full]) != self.storage_path:
            raise ValueError(f"Illegal storage key: {file_name!r}")

        return full


class LocalStorageTTLManager(LocalStorageManager):
    def __init__(self, ttl_seconds: int, sweep_interval_seconds: int, *args, **kwargs):
        if ttl_seconds <= 0:
            raise ValueError("`ttl_seconds` must be greater than or equal to 1.")
        if sweep_interval_seconds <= 0:
            raise ValueError("`sweep_interval_seconds` must be greater than or equal to 1.")

        self._ttl_seconds = ttl_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._sweeper_task: asyncio.Task[None] | None = None

        super().__init__(*args, **kwargs)

    async def save(self, data: bytes, file_name: str) -> SaveContext:
        result = await super().save(data, file_name)
        result.expires_at = result.created_at + self._ttl_seconds
        return result

    async def _sweep_once(self, cutoff: float) -> int:
        expired: list[str] = []
        for entry in os.scandir(self.storage_path):
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    expired.append(entry.path)
            except FileNotFoundError:
                logger.debug("TTL sweep skipped %s; file removed during scan", entry.path)
            except OSError:
                logger.warning("TTL sweep failed to inspect %s", entry.path, exc_info=True)

        deleted = 0
        for path in expired:
            try:
                async with self._io_semaphore:
                    st = await asyncio.to_thread(os.stat, path, follow_symlinks=False)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if st.st_mtime >= cutoff:
                        continue
                    await asyncio.to_thread(os.remove, path)
                    deleted += 1
            except FileNotFoundError:
                logger.debug("TTL sweep skipped %s; file already removed", path)
            except OSError:
                logger.warning("TTL sweep failed to delete expired file %s", path, exc_info=True)

        return deleted

    async def _sweep_loop(self) -> None:
        while True:
            try:
                cutoff = time.time() - self._ttl_seconds
                await self._sweep_once(cutoff)
            except Exception:
                logger.exception("TTL sweep failed for storage path %s", self.storage_path)
            await asyncio.sleep(self._sweep_interval_seconds)

    async def start(self) -> None:
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper_task is None:
            return
        self._sweeper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._sweeper_task
        self._sweeper_task = None


StorageBackendFactory = Callable[[Any], StorageBaseManager]

_STORAGE_BACKENDS: dict[str, StorageBackendFactory] = {}


def register_storage_backend(backend_type: str, factory: StorageBackendFactory) -> None:
    """Register a storage backend under the ``type`` used by the storage config.

    The extension point for artifact offloading: an out-of-tree object-store
    backend (S3, OSS, ...) registers a factory here and returns
    ``UrlStorageHandle`` from ``open``, which the video routes redirect to. No
    object-store client ships in-tree, so nothing here assumes one.
    """
    _STORAGE_BACKENDS[backend_type] = factory


def _build_file_backend(storage_config: FileBackend) -> StorageBaseManager:
    if storage_config.file_ttl is not None and storage_config.ttl_sweep_interval is not None:
        return LocalStorageTTLManager(
            storage_path=storage_config.path,
            max_concurrency=storage_config.file_concurrency,
            ttl_seconds=storage_config.file_ttl,
            sweep_interval_seconds=storage_config.ttl_sweep_interval,
            public_base_url=storage_config.public_base_url,
        )
    return LocalStorageManager(
        storage_path=storage_config.path,
        max_concurrency=storage_config.file_concurrency,
        public_base_url=storage_config.public_base_url,
    )


register_storage_backend("file", _build_file_backend)


def get_storage_manager(storage_config: Any) -> StorageBaseManager:
    backend_type = getattr(storage_config, "type", None)
    factory = _STORAGE_BACKENDS.get(backend_type) if backend_type is not None else None
    if factory is not None:
        return factory(storage_config)

    if isinstance(storage_config, FileBackend):
        if storage_config.file_ttl is not None and storage_config.ttl_sweep_interval is not None:
            manager = LocalStorageTTLManager(
                storage_path=storage_config.path,
                max_concurrency=storage_config.file_concurrency,
                ttl_seconds=storage_config.file_ttl,
                sweep_interval_seconds=storage_config.ttl_sweep_interval,
            )
        else:
            manager = LocalStorageManager(
                storage_path=storage_config.path, max_concurrency=storage_config.file_concurrency
            )
    else:
        raise ValueError("No supported storage managers")

    return manager


STORAGE_MANAGER = get_storage_manager(SERVER_SETTINGS_CONFIG.storage)
