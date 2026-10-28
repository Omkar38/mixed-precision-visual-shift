"""Deterministic M2 image corruptions and split-preserving variants."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter

from mpvs.data.records import AssignedRecord

CORRUPTIONS = ("clean", "darkness", "gaussian_blur")


@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    severity_id: str
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in CORRUPTIONS:
            raise ValueError(f"corruption kind must be one of {CORRUPTIONS}")
        if not self.severity_id.strip():
            raise ValueError("severity_id must be non-empty")
        if self.kind == "clean":
            if self.value is not None:
                raise ValueError("clean corruption must have a null value")
        elif self.value is None or not math.isfinite(self.value):
            raise ValueError(f"{self.kind} requires a finite numeric value")
        elif self.kind == "darkness" and not 0 < self.value < 1:
            raise ValueError("darkness value must be a brightness factor between 0 and 1")
        elif self.kind == "gaussian_blur" and self.value <= 0:
            raise ValueError("gaussian_blur value must be a positive Pillow radius")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CorruptionSpec":
        return cls(**value)

    @property
    def signature(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VariantRecord:
    assigned: AssignedRecord
    corruption: CorruptionSpec

    @property
    def variant_id(self) -> str:
        record = self.assigned.record
        return f"{record.question_id}::{self.corruption.kind}::{self.corruption.severity_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.assigned.to_dict(),
            "variant_id": self.variant_id,
            "corruption": self.corruption.to_dict(),
            "transform_stage": "before_model_processor",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VariantRecord":
        payload = dict(value)
        corruption = CorruptionSpec.from_dict(payload.pop("corruption"))
        payload.pop("variant_id", None)
        transform_stage = payload.pop("transform_stage", None)
        if transform_stage != "before_model_processor":
            raise ValueError(f"unsupported transform stage: {transform_stage!r}")
        return cls(assigned=AssignedRecord.from_dict(payload), corruption=corruption)


def apply_corruption(image: Image.Image, spec: CorruptionSpec) -> Image.Image:
    """Apply a deterministic RGB transform before model-processor resizing."""
    rgb = image.convert("RGB")
    if spec.kind == "clean":
        return rgb.copy()
    if spec.kind == "darkness":
        return ImageEnhance.Brightness(rgb).enhance(spec.value)
    return rgb.filter(ImageFilter.GaussianBlur(radius=spec.value))


def expand_variants(
    records: Iterable[AssignedRecord], specs: Iterable[CorruptionSpec]
) -> tuple[VariantRecord, ...]:
    records = list(records)
    specs = list(specs)
    if not records:
        raise ValueError("cannot expand an empty assigned manifest")
    if not specs:
        raise ValueError("at least one corruption specification is required")
    identities = [(spec.kind, spec.severity_id) for spec in specs]
    if len(set(identities)) != len(identities):
        raise ValueError("corruption kind/severity_id pairs must be unique")
    if sum(spec.kind == "clean" for spec in specs) != 1:
        raise ValueError("variant expansion requires exactly one clean specification")
    return tuple(
        VariantRecord(assigned=assigned, corruption=spec)
        for assigned in sorted(records, key=lambda item: item.record.question_id)
        for spec in sorted(specs, key=lambda item: (item.kind, item.severity_id))
    )


def validate_variant_inheritance(
    base_records: Iterable[AssignedRecord],
    variants: Iterable[VariantRecord],
    specs: Iterable[CorruptionSpec] | None = None,
) -> dict[str, Any]:
    """Check split inheritance and that every question has every paired variant.

    When ``specs`` is omitted, the expected conditions are those observed
    anywhere in ``variants``; passing ``specs`` also catches a condition that
    is missing for every question.
    """
    base = {record.record.question_id: record for record in base_records}
    variants = list(variants)
    observed_specs: dict[tuple[str, str], set[CorruptionSpec]] = defaultdict(set)
    for variant in variants:
        spec = variant.corruption
        observed_specs[(spec.kind, spec.severity_id)].add(spec)
    if specs is None:
        expected = set(observed_specs)
        declared: dict[tuple[str, str], CorruptionSpec] = {}
    else:
        declared = {(spec.kind, spec.severity_id): spec for spec in specs}
        expected = set(declared)
    # The same kind/severity_id must mean the same transform for every question.
    inconsistent_specs = sorted(
        f"{kind}::{severity_id}"
        for (kind, severity_id), seen_specs in observed_specs.items()
        if len(seen_specs) > 1
        or ((kind, severity_id) in declared and seen_specs != {declared[(kind, severity_id)]})
    )
    undeclared_specs = sorted(
        f"{kind}::{severity_id}" for kind, severity_id in set(observed_specs) - expected
    )
    present: dict[str, set[tuple[str, str]]] = defaultdict(set)
    unknown_questions: list[str] = []
    split_mismatches: list[str] = []
    group_mismatches: list[str] = []
    record_mismatches: list[str] = []
    duplicate_variant_ids: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant.variant_id in seen:
            duplicate_variant_ids.append(variant.variant_id)
        seen.add(variant.variant_id)
        question_id = variant.assigned.record.question_id
        present[question_id].add((variant.corruption.kind, variant.corruption.severity_id))
        original = base.get(question_id)
        if original is None:
            unknown_questions.append(question_id)
            continue
        if original.split != variant.assigned.split:
            split_mismatches.append(variant.variant_id)
        if original.record.group_id != variant.assigned.record.group_id:
            group_mismatches.append(variant.variant_id)
        if original.record != variant.assigned.record:
            record_mismatches.append(variant.variant_id)
    missing_variants = sorted(
        f"{question_id}::{kind}::{severity_id}"
        for question_id in base
        for kind, severity_id in expected - present[question_id]
    )
    report = {
        "base_record_count": len(base),
        "variant_count": len(variants),
        "unknown_question_ids": sorted(set(unknown_questions)),
        "split_mismatches": sorted(set(split_mismatches)),
        "group_mismatches": sorted(set(group_mismatches)),
        "record_mismatches": sorted(set(record_mismatches)),
        "duplicate_variant_ids": sorted(set(duplicate_variant_ids)),
        "missing_variants": missing_variants,
        "inconsistent_corruption_specs": inconsistent_specs,
        "undeclared_corruption_specs": undeclared_specs,
    }
    report["valid"] = (
        bool(variants)
        and not unknown_questions
        and not split_mismatches
        and not group_mismatches
        and not record_mismatches
        and not duplicate_variant_ids
        and not missing_variants
        and not inconsistent_specs
        and not undeclared_specs
    )
    return report
