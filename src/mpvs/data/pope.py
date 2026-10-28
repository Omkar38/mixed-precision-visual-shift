"""Strict POPE JSONL normalization without assuming a sampling variant."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from mpvs.data.manifests import read_jsonl
from mpvs.data.records import NormalizedRecord, validate_normalized_records


def _field(row: dict[str, Any], name: str, line_number: int) -> Any:
    if name not in row:
        raise ValueError(f"POPE row {line_number} is missing required field {name!r}")
    return row[name]


def load_pope_jsonl(
    annotation_path: Path,
    *,
    variant: str,
    image_field: str = "image",
    question_field: str = "text",
    label_field: str = "label",
    question_id_field: str = "question_id",
    image_id_field: str | None = None,
) -> list[NormalizedRecord]:
    """Normalize one explicitly selected POPE variant.

    Field names are arguments so Gauri can validate the actual source schema
    instead of the loader silently guessing aliases.
    """
    if not variant.strip():
        raise ValueError("variant must be explicitly named")
    records: list[NormalizedRecord] = []
    for line_number, row in enumerate(read_jsonl(annotation_path), start=1):
        image_relpath = str(_field(row, image_field, line_number)).replace("\\", "/")
        image_id = (
            str(_field(row, image_id_field, line_number))
            if image_id_field is not None
            else PurePosixPath(image_relpath).stem
        )
        records.append(
            NormalizedRecord(
                dataset="pope_coco",
                original_image_id=image_id,
                image_relpath=image_relpath,
                question_id=str(_field(row, question_id_field, line_number)),
                question=str(_field(row, question_field, line_number)),
                label=str(_field(row, label_field, line_number)),
                source_variant=variant.strip(),
            )
        )
    report = validate_normalized_records(records)
    if not report["valid"]:
        raise ValueError(f"invalid normalized POPE records: {report}")
    return records
