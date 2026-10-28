"""CelebA attribute/identity normalization with explicit question mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from mpvs.data.records import NormalizedRecord, validate_normalized_records


def _read_identities(path: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"invalid CelebA identity row at {path}:{line_number}")
            filename, identity_id = parts
            if filename in identities:
                raise ValueError(f"duplicate CelebA identity filename: {filename}")
            identities[filename] = identity_id
    return identities


def load_celeba_annotations(
    attributes_path: Path,
    identities_path: Path,
    *,
    attribute_questions: Mapping[str, str],
) -> list[NormalizedRecord]:
    """Create one yes/no record per selected attribute and source image."""
    if not attribute_questions:
        raise ValueError("at least one explicit CelebA attribute question is required")
    identities = _read_identities(identities_path)
    with attributes_path.open("r", encoding="utf-8") as handle:
        count_line = handle.readline().strip()
        try:
            declared_count = int(count_line)
        except ValueError as exc:
            raise ValueError(f"invalid CelebA image count: {count_line!r}") from exc
        attributes = handle.readline().split()
        if not attributes:
            raise ValueError("CelebA attribute header is empty")
        missing_attributes = sorted(set(attribute_questions) - set(attributes))
        if missing_attributes:
            raise ValueError(f"selected CelebA attributes are absent: {missing_attributes}")
        indexes = {name: attributes.index(name) for name in attribute_questions}
        records: list[NormalizedRecord] = []
        observed_images = 0
        for line_number, line in enumerate(handle, start=3):
            parts = line.split()
            if len(parts) != len(attributes) + 1:
                raise ValueError(f"invalid CelebA attribute row at {attributes_path}:{line_number}")
            filename, raw_values = parts[0], parts[1:]
            if filename not in identities:
                raise ValueError(f"CelebA image {filename!r} has no identity annotation")
            observed_images += 1
            for attribute, question in sorted(attribute_questions.items()):
                raw_label = raw_values[indexes[attribute]]
                if raw_label not in {"-1", "1"}:
                    raise ValueError(
                        f"CelebA attribute {attribute!r} has invalid value {raw_label!r} "
                        f"at {attributes_path}:{line_number}"
                    )
                records.append(
                    NormalizedRecord(
                        dataset="celeba",
                        original_image_id=filename,
                        image_relpath=filename,
                        identity_id=identities[filename],
                        question_id=f"celeba::{filename}::{attribute}",
                        question=question,
                        label="yes" if raw_label == "1" else "no",
                        attribute=attribute,
                    )
                )
    if observed_images != declared_count:
        raise ValueError(
            f"CelebA declared {declared_count} images but parsed {observed_images} attribute rows"
        )
    report = validate_normalized_records(records)
    if not report["valid"]:
        raise ValueError(f"invalid normalized CelebA records: {report}")
    return records
