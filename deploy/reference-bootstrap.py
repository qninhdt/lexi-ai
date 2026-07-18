"""Fetch one immutable Cambridge reference artifact into a shared volume."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

MAX_DATASET_BYTES = 512 * 1024 * 1024


def main() -> None:
    url = required("LEXI_REFERENCE_DATASET_URL")
    expected = required("LEXI_REFERENCE_DATASET_SHA256").lower()
    target = Path(os.environ.get("LEXI_REFERENCE_DATASET_PATH", "/reference/cambridge.db"))
    if urlparse(url).scheme != "https":
        raise ValueError("LEXI_REFERENCE_DATASET_URL must use HTTPS")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("LEXI_REFERENCE_DATASET_SHA256 must be a lowercase SHA-256 digest")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == expected:
        return

    with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
        dir=target.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            downloaded += len(chunk)
            if downloaded > MAX_DATASET_BYTES:
                raise ValueError("reference dataset exceeds the maximum size")
            temporary.write(chunk)
    try:
        if digest(temporary_path) != expected:
            raise ValueError("reference dataset checksum mismatch")
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()
