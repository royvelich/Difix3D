"""Configurable local/S3 input storage for DiFix datasets.

The training code keeps its original filesystem behavior by default.  When
``configure_storage(..., backend="s3_stream")`` is selected, ``s3://`` images
are read directly into memory while files that require a real local path
(manifests, NumPy archives, and memory-mapped depth arrays) are materialized
atomically into a shared on-node cache.

The S3 client is created lazily and per process.  This matters for PyTorch
DataLoader workers: boto clients created in the parent are not fork-safe.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import re
import tarfile
import threading
from typing import Optional
from urllib.parse import urlsplit

from PIL import Image


@dataclass(frozen=True)
class StorageSettings:
    backend: str = "local"
    cache_dir: Optional[str] = None
    cache_images: bool = False
    max_pool_connections: int = 32
    tar_routes_path: Optional[str] = None


_SETTINGS = StorageSettings()
_CLIENT = None
_CLIENT_PID = None
_CLIENT_LOCK = threading.Lock()
_TEST_CLIENT = None
_TAR_ROUTES = ()
_TAR_ROUTE_LOCK = threading.Lock()


class S3StorageError(OSError):
    """An S3 operation failed for reasons other than a missing object."""


def configure_storage(
    *,
    backend: str = "local",
    cache_dir: Optional[str] = None,
    cache_images: bool = False,
    max_pool_connections: int = 32,
    tar_routes_path: Optional[str] = None,
) -> StorageSettings:
    """Configure input storage for the current process and future forked workers."""

    backend = str(backend)
    if backend not in ("local", "s3_stream"):
        raise ValueError(
            f"storage backend must be local|s3_stream, got {backend!r}"
        )
    max_pool_connections = int(max_pool_connections)
    if max_pool_connections < 1:
        raise ValueError("S3 max_pool_connections must be >= 1")
    if cache_dir:
        if is_s3_uri(cache_dir):
            raise ValueError("the S3 materialization cache must be a local path")
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    if tar_routes_path and not is_s3_uri(tar_routes_path):
        tar_routes_path = os.path.abspath(os.path.expanduser(tar_routes_path))
    if tar_routes_path and backend != "s3_stream":
        raise ValueError("tar_routes_path requires backend='s3_stream'")
    if backend == "s3_stream":
        if not cache_dir:
            raise ValueError("s3_stream requires a local cache_dir")
        os.makedirs(cache_dir, exist_ok=True)

    global _SETTINGS, _CLIENT, _CLIENT_PID, _TAR_ROUTES
    _SETTINGS = StorageSettings(
        backend=backend,
        cache_dir=cache_dir,
        cache_images=bool(cache_images),
        max_pool_connections=max_pool_connections,
        tar_routes_path=tar_routes_path,
    )
    # A settings change must not retain a client with an obsolete pool config.
    _CLIENT = None
    _CLIENT_PID = None
    _TAR_ROUTES = _load_tar_routes(tar_routes_path) if tar_routes_path else ()
    return _SETTINGS


def storage_settings() -> StorageSettings:
    return _SETTINGS


def set_s3_client_for_testing(client) -> None:
    """Inject a minimal boto-compatible client.  Tests only."""

    global _TEST_CLIENT
    _TEST_CLIENT = client


def is_s3_uri(path) -> bool:
    return str(path).lower().startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(str(uri))
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError(f"invalid S3 URI: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"S3 URI must not contain query/fragment: {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI must name an object: {uri!r}")
    return parsed.netloc, key


def join_location(root, *parts: object):
    """Join local paths or S3 keys without corrupting the ``s3://`` scheme."""

    root_text = os.fspath(root)
    if is_s3_uri(root_text):
        parsed = urlsplit(root_text)
        key = parsed.path.lstrip("/")
        joined = posixpath.join(
            key.rstrip("/"),
            *(str(part).replace("\\", "/").strip("/") for part in parts),
        )
        return f"s3://{parsed.netloc}/{joined}"
    return os.path.join(root_text, *(os.fspath(part) for part in parts))


def normalize_location(path) -> str:
    text = os.fspath(path)
    if is_s3_uri(text):
        parsed = urlsplit(text)
        key = posixpath.normpath(parsed.path.lstrip("/"))
        return f"s3://{parsed.netloc}/{key}"
    return os.path.realpath(os.path.abspath(os.path.expanduser(text)))


def normalize_logical_path(path) -> str:
    """Normalize path spelling without resolving local symlinks.

    Geometry records compare manifest identities, where preserving the former
    ``os.path.normpath`` behavior is important for local datasets.
    """

    text = os.fspath(path)
    if is_s3_uri(text):
        parsed = urlsplit(text)
        key = posixpath.normpath(parsed.path.lstrip("/"))
        return f"s3://{parsed.netloc}/{key}"
    return os.path.normpath(os.path.expanduser(text))


def location_basename(path) -> str:
    text = os.fspath(path)
    if is_s3_uri(text):
        return posixpath.basename(urlsplit(text).path.rstrip("/"))
    return os.path.basename(os.path.normpath(text))


def location_suffix(path) -> str:
    return os.path.splitext(location_basename(path))[1].lower()


def _require_s3_stream(uri: str) -> None:
    if _SETTINGS.backend != "s3_stream":
        raise RuntimeError(
            f"S3 input {uri!r} requires --storage_backend=s3_stream"
        )


def _s3_client():
    if _TEST_CLIENT is not None:
        return _TEST_CLIENT
    _require_s3_stream("s3://<configured-input>")
    pid = os.getpid()
    global _CLIENT, _CLIENT_PID
    if _CLIENT is not None and _CLIENT_PID == pid:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None and _CLIENT_PID == pid:
            return _CLIENT
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "S3 streaming requires boto3; install the repository requirements "
                "or `pip install boto3` in the DiFix environment"
            ) from exc
        _CLIENT = boto3.client(
            "s3",
            config=Config(
                max_pool_connections=_SETTINGS.max_pool_connections,
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )
        _CLIENT_PID = pid
        return _CLIENT


def _error_status(exc) -> Optional[int]:
    try:
        return int(exc.response["ResponseMetadata"]["HTTPStatusCode"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _read_direct_s3_bytes(
    uri: str,
    *,
    start: Optional[int] = None,
    size: Optional[int] = None,
) -> bytes:
    """Read an S3 object or byte range without applying logical tar routes."""

    bucket, key = parse_s3_uri(uri)
    request = {"Bucket": bucket, "Key": key}
    if start is not None:
        if size is None or size < 0:
            raise ValueError("a ranged S3 read requires a non-negative size")
        if size == 0:
            return b""
        request["Range"] = f"bytes={start}-{start + size - 1}"
    try:
        response = _s3_client().get_object(**request)
    except Exception as exc:
        raise S3StorageError(f"could not stream S3 object {uri}: {exc}") from exc
    body = response["Body"]
    try:
        payload = body.read()
    finally:
        try:
            body.close()
        except Exception:
            pass
    expected = response.get("ContentLength")
    if expected is not None and len(payload) != int(expected):
        raise IOError(
            f"short S3 read for {uri}: got {len(payload)}, expected {expected}"
        )
    if size is not None and len(payload) != size:
        raise IOError(
            f"short ranged S3 read for {uri}: got {len(payload)}, expected {size}"
        )
    return payload


def _read_direct_location(path) -> bytes:
    text = os.fspath(path)
    if is_s3_uri(text):
        return _read_direct_s3_bytes(text)
    return Path(text).read_bytes()


def _load_tar_routes(path) -> tuple[dict, ...]:
    try:
        payload = json.loads(_read_direct_location(path).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"could not load S3 tar routes from {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported S3 tar-route schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    routes = []
    for raw in payload.get("routes", []):
        prefix = normalize_location(raw.get("logical_prefix", "")).rstrip("/")
        mode = str(raw.get("mode", ""))
        inventory_uri = raw.get("inventory_uri")
        if not is_s3_uri(prefix):
            raise ValueError(f"tar-route logical_prefix must be S3: {prefix!r}")
        if mode not in ("scene", "member"):
            raise ValueError(f"tar-route mode must be scene|member, got {mode!r}")
        if not inventory_uri:
            raise ValueError(f"tar route {prefix!r} has no inventory_uri")
        routes.append(
            {
                "prefix": prefix,
                "mode": mode,
                "inventory_uri": os.fspath(inventory_uri),
                "inventory": None,
            }
        )
    routes.sort(key=lambda route: len(route["prefix"]), reverse=True)
    return tuple(routes)


def _route_inventory(route: dict) -> dict:
    inventory = route["inventory"]
    if inventory is not None:
        return inventory
    with _TAR_ROUTE_LOCK:
        inventory = route["inventory"]
        if inventory is not None:
            return inventory
        try:
            payload = json.loads(
                _read_direct_location(route["inventory_uri"]).decode("utf-8")
            )
        except Exception as exc:
            raise S3StorageError(
                f"could not load tar inventory {route['inventory_uri']}: {exc}"
            ) from exc
        if payload.get("schema_version", 1) != 1:
            raise S3StorageError(
                f"unsupported tar-inventory schema in {route['inventory_uri']}: "
                f"{payload.get('schema_version')!r}"
            )
        if route["mode"] == "scene":
            archives = {
                str(entry["scene"]): str(entry["archive_uri"])
                for entry in payload.get("entries", [])
                if entry.get("scene") and entry.get("archive_uri")
            }
            inventory = {
                "archives": archives,
                "scene_lengths": tuple(
                    sorted({len(scene) for scene in archives}, reverse=True)
                ),
            }
        else:
            archives = {}
            for entry in payload.get("entries", []):
                archive_uri = entry.get("archive_uri")
                if not archive_uri:
                    continue
                for member in entry.get("members", []):
                    archives[str(member).lstrip("/")] = str(archive_uri)
            inventory = {"archives": archives}
        route["inventory"] = inventory
        return inventory


def _tar_target(path) -> Optional[tuple[str, str]]:
    """Resolve a logical S3 path to ``(archive_uri, member_name)``."""

    text = normalize_location(path)
    for route in _TAR_ROUTES:
        prefix = route["prefix"]
        if text != prefix and not text.startswith(prefix + "/"):
            continue
        member = text[len(prefix):].lstrip("/")
        if not member:
            return None
        inventory = _route_inventory(route)
        if route["mode"] == "member":
            archive_uri = inventory["archives"].get(member)
        else:
            archive_uri = None
            for component in member.split("/"):
                for length in inventory["scene_lengths"]:
                    if len(component) < length:
                        continue
                    archive_uri = inventory["archives"].get(component[:length])
                    if archive_uri is not None:
                        break
                if archive_uri is not None:
                    break
        if archive_uri is None:
            # Files such as data.json can live directly under a broad routed
            # prefix while scene data lives inside archives.
            return None
        return archive_uri, member
    return None


class _S3RangeReader(io.RawIOBase):
    """Small seekable facade used by tarfile to index remote tar files."""

    _CHUNK_SIZE = 64 * 1024

    def __init__(self, uri: str, size: int):
        self.uri = uri
        self.size = int(size)
        self.position = 0
        self._chunk_start = -1
        self._chunk = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self.position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid seek whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = int(position)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        remaining = min(int(size), self.size - self.position)
        if remaining <= 0:
            return b""
        chunks = []
        while remaining:
            chunk_start = (self.position // self._CHUNK_SIZE) * self._CHUNK_SIZE
            if chunk_start != self._chunk_start:
                chunk_size = min(self._CHUNK_SIZE, self.size - chunk_start)
                self._chunk = _read_direct_s3_bytes(
                    self.uri, start=chunk_start, size=chunk_size
                )
                self._chunk_start = chunk_start
            offset = self.position - self._chunk_start
            take = min(remaining, len(self._chunk) - offset)
            if take <= 0:
                raise IOError(f"could not advance ranged reader for {self.uri}")
            chunks.append(self._chunk[offset:offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def _tar_index_path(archive_uri: str) -> Path:
    if not _SETTINGS.cache_dir:
        raise RuntimeError("S3 materialization cache is not configured")
    digest = hashlib.sha256(archive_uri.encode("utf-8")).hexdigest()
    return Path(_SETTINGS.cache_dir) / "tar_indexes" / f"{digest}.json"


def _load_tar_index(archive_uri: str, metadata: dict) -> dict:
    index_path = _tar_index_path(archive_uri)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    def read_valid_index():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if (
            payload.get("etag") == metadata.get("etag")
            and int(payload.get("size", -1)) == int(metadata["size"])
        ):
            return payload.get("members", {})
        return None

    members = read_valid_index()
    if members is not None:
        return members
    with _exclusive_lock(Path(f"{index_path}.lock")):
        members = read_valid_index()
        if members is not None:
            return members
        reader = _S3RangeReader(archive_uri, metadata["size"])
        try:
            with tarfile.open(fileobj=reader, mode="r:") as archive:
                members = {
                    member.name.lstrip("/"): [member.offset_data, member.size]
                    for member in archive
                    if member.isfile()
                }
        except Exception as exc:
            raise S3StorageError(
                f"could not index uncompressed tar archive {archive_uri}: {exc}"
            ) from exc
        payload = {
            "schema_version": 1,
            "archive_uri": archive_uri,
            "etag": metadata.get("etag"),
            "size": metadata["size"],
            "members": members,
        }
        temporary = Path(
            f"{index_path}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        temporary.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, index_path)
        return members


def _tar_member_spec(path) -> Optional[dict]:
    target = _tar_target(path)
    if target is None:
        return None
    archive_uri, member = target
    metadata = _object_metadata(archive_uri)
    indexed = _load_tar_index(archive_uri, metadata).get(member)
    if indexed is None:
        return {
            "archive_uri": archive_uri,
            "member": member,
            "missing": True,
        }
    offset, size = map(int, indexed)
    return {
        "archive_uri": archive_uri,
        "member": member,
        "offset": offset,
        "size": size,
        "etag": metadata.get("etag"),
    }


def _read_tar_member(spec: dict) -> bytes:
    if spec.get("missing"):
        raise S3StorageError(
            f"member {spec['member']!r} is absent from {spec['archive_uri']}"
        )
    return _read_direct_s3_bytes(
        spec["archive_uri"], start=spec["offset"], size=spec["size"]
    )


def location_exists(path) -> bool:
    text = os.fspath(path)
    if not is_s3_uri(text):
        return os.path.exists(text)
    _require_s3_stream(text)
    spec = _tar_member_spec(text)
    if spec is not None:
        return not spec.get("missing", False)
    bucket, key = parse_s3_uri(text)
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        if _error_status(exc) == 404:
            return False
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if str(code) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise S3StorageError(f"could not inspect S3 object {text}: {exc}") from exc


def _object_metadata(uri: str) -> dict:
    bucket, key = parse_s3_uri(uri)
    try:
        head = _s3_client().head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise S3StorageError(
            f"could not inspect S3 object {uri}: {exc}"
        ) from exc
    return {
        "etag": str(head.get("ETag", "")).strip('"'),
        "size": int(head["ContentLength"]),
    }


def _cache_path(uri: str) -> Path:
    if not _SETTINGS.cache_dir:
        raise RuntimeError("S3 materialization cache is not configured")
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", location_basename(uri))
    if not basename:
        basename = "object"
    return (
        Path(_SETTINGS.cache_dir)
        / "objects"
        / digest[:2]
        / f"{digest}-{basename}"
    )


@contextmanager
def _exclusive_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _cached_metadata(path: Path) -> Optional[dict]:
    metadata_path = Path(f"{path}.s3.json")
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _cache_matches(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    actual = _cached_metadata(path)
    return (
        actual is not None
        and actual.get("etag") == expected.get("etag")
        and int(actual.get("size", -1)) == int(expected.get("size", -2))
        and path.stat().st_size == int(expected["size"])
    )


def _stream_object_to_file(uri: str, destination: Path, expected: dict) -> None:
    bucket, key = parse_s3_uri(uri)
    request = {"Bucket": bucket, "Key": key}
    if expected.get("etag"):
        request["IfMatch"] = f'"{expected["etag"]}"'
    try:
        response = _s3_client().get_object(**request)
    except Exception as exc:
        raise S3StorageError(
            f"could not download S3 object {uri}: {exc}"
        ) from exc
    body = response["Body"]
    temporary = Path(
        f"{destination}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    written = 0
    try:
        with temporary.open("wb") as handle:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if written != int(expected["size"]):
            raise IOError(
                f"short S3 read for {uri}: got {written}, expected {expected['size']}"
            )
        os.replace(temporary, destination)
    finally:
        try:
            body.close()
        except Exception:
            pass
        if temporary.exists():
            temporary.unlink()


def _write_cache_metadata(path: Path, metadata: dict) -> None:
    metadata_path = Path(f"{path}.s3.json")
    temporary = Path(
        f"{metadata_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)


def materialize(path, *, validate: bool = True) -> str:
    """Return a local path, downloading an S3 object atomically when needed."""

    text = os.fspath(path)
    if not is_s3_uri(text):
        return os.path.abspath(os.path.expanduser(text))
    _require_s3_stream(text)
    spec = _tar_member_spec(text)
    if spec is not None:
        if spec.get("missing"):
            raise S3StorageError(
                f"member {spec['member']!r} is absent from {spec['archive_uri']}"
            )
        destination = _cache_path(text)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "etag": (
                f"{spec.get('etag', '')}:{spec['offset']}:{spec['size']}"
            ),
            "size": spec["size"],
        }
        if _cache_matches(destination, metadata):
            return str(destination)
        with _exclusive_lock(Path(f"{destination}.lock")):
            if not _cache_matches(destination, metadata):
                temporary = Path(
                    f"{destination}.tmp.{os.getpid()}.{threading.get_ident()}"
                )
                bucket, key = parse_s3_uri(spec["archive_uri"])
                request = {
                    "Bucket": bucket,
                    "Key": key,
                    "Range": (
                        f"bytes={spec['offset']}-"
                        f"{spec['offset'] + spec['size'] - 1}"
                    ),
                }
                if spec.get("etag"):
                    request["IfMatch"] = f'"{spec["etag"]}"'
                try:
                    response = _s3_client().get_object(**request)
                except Exception as exc:
                    raise S3StorageError(
                        f"could not extract {spec['member']!r} from "
                        f"{spec['archive_uri']}: {exc}"
                    ) from exc
                body = response["Body"]
                written = 0
                try:
                    with temporary.open("wb") as handle:
                        while True:
                            chunk = body.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            written += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if written != spec["size"]:
                        raise IOError(
                            f"short tar-member read for {text}: got {written}, "
                            f"expected {spec['size']}"
                        )
                    os.replace(temporary, destination)
                    _write_cache_metadata(destination, metadata)
                finally:
                    try:
                        body.close()
                    except Exception:
                        pass
                    if temporary.exists():
                        temporary.unlink()
        return str(destination)
    destination = _cache_path(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _object_metadata(text)
    if _cache_matches(destination, metadata):
        return str(destination)
    lock_path = Path(f"{destination}.lock")
    with _exclusive_lock(lock_path):
        # Another DDP rank or DataLoader worker may have completed the object
        # while this process was waiting for the lock.
        if validate:
            metadata = _object_metadata(text)
        if not _cache_matches(destination, metadata):
            # A process killed during a previous download cannot execute its
            # finally block. The per-object lock makes these stale siblings
            # safe to remove before starting the next attempt.
            for stale in destination.parent.glob(f"{destination.name}.tmp.*"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            _stream_object_to_file(text, destination, metadata)
            _write_cache_metadata(destination, metadata)
    return str(destination)


def read_bytes(path, *, cache: Optional[bool] = None) -> bytes:
    """Read a local object or stream one S3 object into memory."""

    text = os.fspath(path)
    if not is_s3_uri(text):
        return Path(text).read_bytes()
    _require_s3_stream(text)
    if cache is None:
        cache = _SETTINGS.cache_images
    spec = _tar_member_spec(text)
    if spec is not None:
        if cache:
            return Path(materialize(text, validate=False)).read_bytes()
        return _read_tar_member(spec)
    if cache:
        return Path(materialize(text, validate=False)).read_bytes()
    return _read_direct_s3_bytes(text)


def read_text(path, *, encoding: str = "utf-8") -> str:
    text = os.fspath(path)
    if is_s3_uri(text):
        return read_bytes(text, cache=True).decode(encoding)
    return Path(text).read_text(encoding=encoding)


def load_pil_image(path, *, mode: Optional[str] = None) -> Image.Image:
    """Load an independent PIL image from local disk or S3."""

    text = os.fspath(path)
    if is_s3_uri(text):
        source = io.BytesIO(read_bytes(text))
        with source, Image.open(source) as image:
            return image.convert(mode) if mode else image.copy()
    with Image.open(text) as image:
        return image.convert(mode) if mode else image.copy()


def pil_image_size(path) -> tuple[int, int]:
    """Read image dimensions without decoding the full image when possible."""

    text = os.fspath(path)
    if not is_s3_uri(text):
        with Image.open(text) as image:
            return image.size
    _require_s3_stream(text)
    if _SETTINGS.cache_images:
        with Image.open(materialize(text, validate=False)) as image:
            return image.size
    if _tar_target(text) is not None:
        return load_pil_image(text).size

    bucket, key = parse_s3_uri(text)
    # PNG/JPEG/WebP dimensions normally appear in the first few KiB. Retry
    # with progressively larger prefixes for images carrying large metadata,
    # then retain the full-read fallback for unusual formats.
    for prefix_size in (64 * 1024, 256 * 1024, 1024 * 1024):
        try:
            response = _s3_client().get_object(
                Bucket=bucket,
                Key=key,
                Range=f"bytes=0-{prefix_size - 1}",
            )
        except Exception as exc:
            raise S3StorageError(
                f"could not read S3 image header {text}: {exc}"
            ) from exc
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            try:
                body.close()
            except Exception:
                pass
        try:
            with Image.open(io.BytesIO(payload)) as image:
                return image.size
        except (OSError, ValueError):
            continue
    return load_pil_image(text).size
