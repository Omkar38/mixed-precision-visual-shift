"""Deterministic, group-disjoint M2 split assignment and validation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from mpvs.data.records import AssignedRecord, NormalizedRecord, SPLITS, validate_normalized_records


@dataclass(frozen=True)
class SplitSpec:
    seed: int
    pilot_fraction: float
    development_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        values = self.fractions
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("split fractions must be finite and nonnegative")
        if not math.isclose(sum(values.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("split fractions must sum to 1")
        if sum(value > 0 for value in values.values()) < 2:
            raise ValueError("at least two split fractions must be positive")

    @property
    def fractions(self) -> dict[str, float]:
        return {
            "pilot": self.pilot_fraction,
            "development": self.development_fraction,
            "test": self.test_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _group_order(dataset: str, group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{dataset}\0{group_id}".encode()).hexdigest()


def _group_counts(group_count: int, fractions: dict[str, float]) -> dict[str, int]:
    positive = [split for split in SPLITS if fractions[split] > 0]
    if group_count < len(positive):
        raise ValueError(
            f"need at least {len(positive)} groups to populate every requested split; "
            f"found {group_count}"
        )
    raw = {split: group_count * fractions[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = group_count - sum(counts.values())
    ranked = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - math.floor(raw[split])), SPLITS.index(split)),
    )
    for split in ranked[:remaining]:
        counts[split] += 1
    # A small fixture can round a positive fraction to zero. Preserve the
    # requested three-way contract by moving one group from the most
    # over-allocated split; whole-group integrity still takes precedence.
    for split in positive:
        if counts[split] > 0:
            continue
        donors = [candidate for candidate in positive if counts[candidate] > 1]
        donor = max(
            donors,
            key=lambda candidate: (
                counts[candidate] - raw[candidate],
                counts[candidate],
                -SPLITS.index(candidate),
            ),
        )
        counts[donor] -= 1
        counts[split] = 1
    return counts


def split_records(
    records: Iterable[NormalizedRecord], spec: SplitSpec
) -> dict[str, tuple[AssignedRecord, ...]]:
    """Assign indivisible image/identity groups independent of input ordering."""
    records = list(records)
    validation = validate_normalized_records(records)
    if not validation["valid"]:
        raise ValueError(f"records are not valid for splitting: {validation}")
    if len(validation["datasets"]) != 1:
        raise ValueError("split one dataset at a time so its group ratios remain explicit")
    dataset = records[0].dataset
    groups: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    ordered_groups = sorted(groups, key=lambda group_id: _group_order(dataset, group_id, spec.seed))
    counts = _group_counts(len(ordered_groups), spec.fractions)
    group_splits: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        for group_id in ordered_groups[offset : offset + counts[split]]:
            group_splits[group_id] = split
        offset += counts[split]
    assigned = {split: [] for split in SPLITS}
    for record in sorted(records, key=lambda item: item.question_id):
        split = group_splits[record.group_id]
        assigned[split].append(AssignedRecord(record=record, split=split))
    result = {split: tuple(assigned[split]) for split in SPLITS}
    report = validate_assigned_records(item for rows in result.values() for item in rows)
    if not report["valid"]:
        raise AssertionError(f"internal split validation failed: {report}")
    return result


def validate_assigned_records(records: Iterable[AssignedRecord]) -> dict[str, Any]:
    records = list(records)
    question_counts: Counter[str] = Counter()
    question_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    identity_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts = {split: {"records": 0, "groups": set(), "labels": Counter()} for split in SPLITS}
    for assigned in records:
        record = assigned.record
        question_counts[record.question_id] += 1
        question_splits[record.question_id].add(assigned.split)
        image_splits[(record.dataset, record.original_image_id)].add(assigned.split)
        group_splits[(record.dataset, record.group_id)].add(assigned.split)
        if record.dataset == "celeba":
            identity_splits[record.identity_id or ""].add(assigned.split)
        counts[assigned.split]["records"] += 1
        counts[assigned.split]["groups"].add(record.group_id)
        counts[assigned.split]["labels"][record.label] += 1
    duplicate_question_ids = sorted(
        question_id for question_id, count in question_counts.items() if count > 1
    )
    question_split_conflicts = sorted(
        question_id for question_id, splits in question_splits.items() if len(splits) > 1
    )
    image_leakage = sorted(
        f"{dataset}:{image_id}"
        for (dataset, image_id), splits in image_splits.items()
        if len(splits) > 1
    )
    identity_leakage = sorted(
        identity_id for identity_id, splits in identity_splits.items() if len(splits) > 1
    )
    group_leakage = sorted(
        f"{dataset}:{group_id}"
        for (dataset, group_id), splits in group_splits.items()
        if len(splits) > 1
    )
    report = {
        "record_count": len(records),
        "duplicate_question_ids": duplicate_question_ids,
        "question_id_split_conflicts": question_split_conflicts,
        "original_image_leakage": image_leakage,
        "celeba_identity_leakage": identity_leakage,
        "group_leakage": group_leakage,
        "splits": {
            split: {
                "record_count": details["records"],
                "group_count": len(details["groups"]),
                "label_counts": dict(sorted(details["labels"].items())),
            }
            for split, details in counts.items()
        },
    }
    report["valid"] = (
        bool(records)
        and not duplicate_question_ids
        and not question_split_conflicts
        and not image_leakage
        and not identity_leakage
        and not group_leakage
    )
    return report
