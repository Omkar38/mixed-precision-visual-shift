"""Deterministic, atomic JSON/JSONL manifest I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _immutable_text(path: Path, content: str) -> str:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to replace immutable manifest with different content: {path}"
            )
        return sha256_file(path)
    _atomic_text(path, content)
    return sha256_file(path)


def write_json_atomic(path: Path, value: Any) -> str:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def write_json_immutable(path: Path, value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return _immutable_text(path, content)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    encoded = sorted(canonical_json(row) for row in rows)
    _atomic_text(path, "".join(f"{row}\n" for row in encoded))
    return sha256_file(path)


def write_jsonl_immutable(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    encoded = sorted(canonical_json(row) for row in rows)
    return _immutable_text(path, "".join(f"{row}\n" for row in encoded))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows
