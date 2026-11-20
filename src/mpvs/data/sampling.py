"""Deterministic, balanced CelebA sampling before identity-group splitting."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from mpvs.data.records import LABELS, NormalizedRecord, validate_normalized_records


@dataclass(frozen=True)
class CelebASamplingSpec:
    """Explicit inputs for one balanced CelebA sampling run."""

    seed: int
    attributes: tuple[str, ...]
    per_label_count: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("sampling seed must be an integer")
        cleaned = tuple(sorted(attribute.strip() for attribute in self.attributes))
        if not cleaned or any(not attribute for attribute in cleaned):
            raise ValueError("at least one non-empty CelebA attribute is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("CelebA sampling attributes must be unique")
        if (
            isinstance(self.per_label_count, bool)
            or not isinstance(self.per_label_count, int)
            or self.per_label_count <= 0
        ):
            raise ValueError("per_label_count must be a positive integer")
        object.__setattr__(self, "attributes", cleaned)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["attributes"] = list(self.attributes)
        value["one_question_per_image"] = True
        value["one_image_per_identity_per_attribute"] = True
        return value


def _record_rank(spec: CelebASamplingSpec, record: NormalizedRecord) -> str:
    payload = "\0".join(
        (
            str(spec.seed),
            record.attribute or "",
            record.label,
            record.identity_id or "",
            record.original_image_id,
            record.question_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_celeba_records(
    records: Iterable[NormalizedRecord], spec: CelebASamplingSpec
) -> tuple[NormalizedRecord, ...]:
    """Select an exact balanced sample without image or attribute-identity reuse.

    Attributes are processed in sorted order. Within each attribute, the label
    with fewer available identities is selected first. This protects a rare
    positive label from being displaced by the abundant negative label.
    """

    records = list(records)
    source_report = validate_normalized_records(records)
    if not source_report["valid"]:
        raise ValueError(f"source records are invalid: {source_report}")
    if source_report["datasets"] != ["celeba"]:
        raise ValueError("CelebA sampling accepts only normalized CelebA records")

    available_attributes = {record.attribute for record in records}
    missing_attributes = sorted(set(spec.attributes) - available_attributes)
    if missing_attributes:
        raise ValueError(f"requested CelebA attributes are absent: {missing_attributes}")

    selected: list[NormalizedRecord] = []
    used_images: set[str] = set()
    used_checksums: set[str] = set()

    for attribute in spec.attributes:
        attribute_records = [record for record in records if record.attribute == attribute]
        identities_by_label = {
            label: {record.identity_id for record in attribute_records if record.label == label}
            for label in LABELS
        }
        label_order = sorted(LABELS, key=lambda label: (len(identities_by_label[label]), label))
        used_identities: set[str] = set()

        for label in label_order:
            candidates = sorted(
                (record for record in attribute_records if record.label == label),
                key=lambda record: (_record_rank(spec, record), record.question_id),
            )
            chosen = 0
            for record in candidates:
                checksum = record.source_image_sha256
                if record.identity_id in used_identities:
                    continue
                if record.original_image_id in used_images:
                    continue
                if checksum is not None and checksum in used_checksums:
                    continue
                selected.append(record)
                used_identities.add(record.identity_id or "")
                used_images.add(record.original_image_id)
                if checksum is not None:
                    used_checksums.add(checksum)
                chosen += 1
                if chosen == spec.per_label_count:
                    break
            if chosen != spec.per_label_count:
                raise ValueError(
                    "cannot build requested CelebA sample for "
                    f"attribute={attribute!r}, label={label!r}: requested "
                    f"{spec.per_label_count}, selected {chosen}; reduce the requested "
                    "count or inspect identity/image conflicts"
                )

    result = tuple(sorted(selected, key=lambda record: record.question_id))
    report = validate_celeba_sample(result, spec)
    if not report["valid"]:
        raise AssertionError(f"internal CelebA sample validation failed: {report}")
    return result


def validate_celeba_sample(
    records: Iterable[NormalizedRecord],
    spec: CelebASamplingSpec,
    *,
    require_image_checksums: bool = False,
) -> dict[str, Any]:
    """Report exact balance and the sampler's identity/image guarantees."""

    records = list(records)
    normalized = validate_normalized_records(
        records, require_image_checksums=require_image_checksums
    )
    counts: Counter[tuple[str | None, str]] = Counter(
        (record.attribute, record.label) for record in records
    )
    image_counts = Counter(record.original_image_id for record in records)
    identity_attribute_counts = Counter(
        (record.attribute, record.identity_id) for record in records
    )
    checksum_images: dict[str, set[str]] = defaultdict(set)
    questions_by_attribute: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.source_image_sha256 is not None:
            checksum_images[record.source_image_sha256].add(record.original_image_id)
        questions_by_attribute[record.attribute or ""].add(record.question)

    unexpected_attributes = sorted(
        {record.attribute or "" for record in records} - set(spec.attributes)
    )
    missing_attributes = sorted(
        set(spec.attributes) - {record.attribute or "" for record in records}
    )
    incorrect_counts = {
        attribute: {
            label: counts[(attribute, label)]
            for label in LABELS
            if counts[(attribute, label)] != spec.per_label_count
        }
        for attribute in spec.attributes
    }
    incorrect_counts = {
        attribute: values for attribute, values in incorrect_counts.items() if values
    }
    duplicate_images = sorted(image for image, count in image_counts.items() if count > 1)
    repeated_identity_attributes = sorted(
        f"{attribute}:{identity}"
        for (attribute, identity), count in identity_attribute_counts.items()
        if count > 1
    )
    duplicate_checksum_groups = sorted(
        (
            {
                "sha256": checksum,
                "image_ids": sorted(image_ids),
            }
            for checksum, image_ids in checksum_images.items()
            if len(image_ids) > 1
        ),
        key=lambda value: value["sha256"],
    )
    inconsistent_questions = {
        attribute: sorted(questions)
        for attribute, questions in questions_by_attribute.items()
        if len(questions) > 1
    }
    report = {
        "record_count": len(records),
        "requested": spec.to_dict(),
        "attribute_label_counts": {
            attribute: {label: counts[(attribute, label)] for label in LABELS}
            for attribute in spec.attributes
        },
        "distinct_images": len(image_counts),
        "distinct_identities": len({record.identity_id for record in records}),
        "missing_attributes": missing_attributes,
        "unexpected_attributes": unexpected_attributes,
        "incorrect_attribute_label_counts": incorrect_counts,
        "duplicate_original_image_ids": duplicate_images,
        "repeated_identity_attribute_pairs": repeated_identity_attributes,
        "duplicate_source_checksum_groups": duplicate_checksum_groups,
        "inconsistent_questions_by_attribute": inconsistent_questions,
        "missing_source_image_checksums": normalized["missing_source_image_checksums"],
        "normalized_record_validation": normalized,
    }
    report["valid"] = (
        normalized["valid"]
        and not missing_attributes
        and not unexpected_attributes
        and not incorrect_counts
        and not duplicate_images
        and not repeated_identity_attributes
        and not duplicate_checksum_groups
        and not inconsistent_questions
    )
    return report
