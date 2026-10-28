"""Source-image verification shared by M2 dataset loaders."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from mpvs.data.manifests import sha256_file
from mpvs.data.records import NormalizedRecord


def attach_source_image_checksums(
    records: Iterable[NormalizedRecord], image_root: Path
) -> list[NormalizedRecord]:
    """Verify every referenced image and attach a cached SHA-256 digest."""
    records = list(records)
    digests: dict[str, str] = {}
    result: list[NormalizedRecord] = []
    for record in records:
        if record.image_relpath not in digests:
            image_path = image_root / record.image_relpath
            if not image_path.is_file():
                raise FileNotFoundError(f"referenced image does not exist: {image_path}")
            digests[record.image_relpath] = sha256_file(image_path)
        result.append(replace(record, source_image_sha256=digests[record.image_relpath]))
    return result
