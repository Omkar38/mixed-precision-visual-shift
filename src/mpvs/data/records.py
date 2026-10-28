"""Normalized M2 records shared by POPE-COCO and CelebA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable

DATASETS = ("pope_coco", "celeba")
LABELS = ("yes", "no")
SPLITS = ("pilot", "development", "test")


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_posix_path(value: str) -> str:
    value = _required_text("image_relpath", value).replace("\\", "/")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("image_relpath must be a safe relative path")
    return candidate.as_posix()


@dataclass(frozen=True)
class NormalizedRecord:
    dataset: str
    original_image_id: str
    image_relpath: str
    question_id: str
    question: str
    label: str
    identity_id: str | None = None
    source_variant: str | None = None
    attribute: str | None = None
    source_image_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        object.__setattr__(
            self, "original_image_id", _required_text("original_image_id", self.original_image_id)
        )
        object.__setattr__(self, "image_relpath", _relative_posix_path(self.image_relpath))
        object.__setattr__(self, "question_id", _required_text("question_id", self.question_id))
        object.__setattr__(self, "question", _required_text("question", self.question))
        normalized_label = _required_text("label", self.label).lower()
        if normalized_label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        object.__setattr__(self, "label", normalized_label)
        if self.dataset == "celeba":
            object.__setattr__(
                self, "identity_id", _required_text("identity_id", self.identity_id or "")
            )
            object.__setattr__(self, "attribute", _required_text("attribute", self.attribute or ""))
            if self.source_variant is not None:
                raise ValueError("source_variant must be null for CelebA")
        elif self.identity_id is not None:
            raise ValueError("identity_id must be null for POPE-COCO")
        else:
            object.__setattr__(
                self, "source_variant", _required_text("source_variant", self.source_variant or "")
            )
            if self.attribute is not None:
                raise ValueError("attribute must be null for POPE-COCO")
        if self.source_image_sha256 is not None:
            digest = self.source_image_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("source_image_sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "source_image_sha256", digest)

    @property
    def group_id(self) -> str:
        """Return the indivisible split group required by the protocol."""
        return self.identity_id if self.dataset == "celeba" else self.original_image_id

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizedRecord":
        payload = dict(value)
        schema_version = payload.pop("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported normalized-record schema: {schema_version}")
        payload.pop("split", None)
        return cls(**payload)


@dataclass(frozen=True)
class AssignedRecord:
    record: NormalizedRecord
    split: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")

    def to_dict(self) -> dict[str, Any]:
        return {**self.record.to_dict(), "split": self.split}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AssignedRecord":
        payload = dict(value)
        split = payload.pop("split")
        return cls(record=NormalizedRecord.from_dict(payload), split=split)


def validate_normalized_records(
    records: Iterable[NormalizedRecord], *, require_image_checksums: bool = False
) -> dict[str, Any]:
    records = list(records)
    duplicate_question_ids: list[str] = []
    seen_questions: set[str] = set()
    image_identity: dict[str, str] = {}
    image_paths: dict[tuple[str, str], str] = {}
    image_checksums: dict[tuple[str, str], str | None] = {}
    conflicting_image_identities: list[str] = []
    conflicting_image_paths: list[str] = []
    conflicting_image_checksums: list[str] = []
    for record in records:
        if record.question_id in seen_questions:
            duplicate_question_ids.append(record.question_id)
        seen_questions.add(record.question_id)
        image_key = (record.dataset, record.original_image_id)
        previous_path = image_paths.setdefault(image_key, record.image_relpath)
        if previous_path != record.image_relpath:
            conflicting_image_paths.append(f"{record.dataset}:{record.original_image_id}")
        if image_key not in image_checksums:
            image_checksums[image_key] = record.source_image_sha256
        elif image_checksums[image_key] is None and record.source_image_sha256 is not None:
            image_checksums[image_key] = record.source_image_sha256
        elif (
            image_checksums[image_key] is not None
            and record.source_image_sha256 is not None
            and image_checksums[image_key] != record.source_image_sha256
        ):
            conflicting_image_checksums.append(f"{record.dataset}:{record.original_image_id}")
        if record.dataset == "celeba":
            previous = image_identity.setdefault(record.original_image_id, record.identity_id or "")
            if previous != record.identity_id:
                conflicting_image_identities.append(record.original_image_id)
    missing_checksums = [
        record.question_id for record in records if record.source_image_sha256 is None
    ]
    datasets = sorted({record.dataset for record in records})
    report = {
        "record_count": len(records),
        "datasets": datasets,
        "duplicate_question_ids": sorted(set(duplicate_question_ids)),
        "conflicting_celeba_image_identities": sorted(set(conflicting_image_identities)),
        "conflicting_image_paths": sorted(set(conflicting_image_paths)),
        "conflicting_source_image_checksums": sorted(set(conflicting_image_checksums)),
        "missing_source_image_checksums": sorted(missing_checksums),
    }
    report["valid"] = (
        bool(records)
        and not report["duplicate_question_ids"]
        and not report["conflicting_celeba_image_identities"]
        and not report["conflicting_image_paths"]
        and not report["conflicting_source_image_checksums"]
        and (not require_image_checksums or not missing_checksums)
    )
    return report
