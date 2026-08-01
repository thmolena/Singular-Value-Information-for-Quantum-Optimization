"""Authenticated access to the official Stanford Gset inputs.

The project intentionally does not redistribute Gset bytes.  Callers may pass
an explicit path, set an environment variable, or opt into a download from the
fixed Stanford HTTPS endpoint.  Every path is accepted only after a SHA-256
check.
"""

from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    sha256: str
    environment_variable: str


SOURCES = {
    "G1": DataSource(
        name="G1",
        url="https://web.stanford.edu/~yyye/yyye/Gset/G1",
        sha256="73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
        environment_variable="HQML_G1_PATH",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path, source: DataSource) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{source.name} input does not exist: {resolved}")
    actual = sha256(resolved)
    if actual != source.sha256:
        raise ValueError(
            f"{source.name} SHA-256 mismatch: {actual}; expected {source.sha256}"
        )
    return resolved


def _cache_root() -> Path:
    override = os.environ.get("HQML_DATA_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "hqml_spectral_qaoa"


def authenticated_gset(
    name: str,
    *,
    explicit: str | Path | None = None,
    fetch: bool = False,
) -> Path:
    """Resolve a checksum-verified Gset file without bundling source bytes."""

    try:
        source = SOURCES[name]
    except KeyError as error:
        raise ValueError(f"unsupported Gset input: {name}") from error

    if explicit is not None:
        return _validate(Path(explicit), source)
    environment_path = os.environ.get(source.environment_variable)
    if environment_path:
        return _validate(Path(environment_path), source)

    cached = _cache_root() / source.name
    if cached.is_file():
        return _validate(cached, source)
    if not fetch:
        raise FileNotFoundError(
            f"{source.name} is not bundled. Pass --fetch-data, --{name.lower()}-path, "
            f"or set {source.environment_variable}."
        )

    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_name(f".{cached.name}.{os.getpid()}.part")
    parsed = urllib.parse.urlparse(source.url)
    if parsed.scheme != "https" or parsed.hostname != "web.stanford.edu":
        raise ValueError("data source must be the registered Stanford HTTPS host")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "hqml-spectral-qaoa/1.0 data fetch"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            if response.status != 200:
                raise OSError(f"download returned HTTP {response.status}")
            with temporary.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
        _validate(temporary, source)
        temporary.replace(cached)
    finally:
        temporary.unlink(missing_ok=True)
    return _validate(cached, source)
