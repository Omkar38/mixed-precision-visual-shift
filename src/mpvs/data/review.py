"""Model-independent pilot answerability review contracts for M2."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from mpvs.corruptions import VariantRecord


@dataclass(frozen=True)
class AnswerabilityDecision:
    variant_id: str
    answerable: bool
    review_method: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("variant_id must be non-empty")
        if not self.review_method.strip():
            raise ValueError("review_method must be non-empty")
        if not isinstance(self.answerable, bool):
            raise ValueError("answerable must be a JSON boolean")
        if not self.answerable and (self.reason is None or not self.reason.strip()):
            raise ValueError("unanswerable decisions require a reason")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnswerabilityDecision":
        payload = dict(value)
        schema_version = payload.pop("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported answerability schema: {schema_version}")
        return cls(**payload)


def validate_answerability_decisions(
    variants: Iterable[VariantRecord], decisions: Iterable[AnswerabilityDecision]
) -> dict[str, Any]:
    """Require one model-independent decision per corrupted pilot variant."""
    variants = list(variants)
    decisions = list(decisions)
    expected = {
        variant.variant_id: variant
        for variant in variants
        if variant.corruption.kind != "clean"
    }
    counts = Counter(decision.variant_id for decision in decisions)
    observed = set(counts)
    non_pilot_variants = sorted(
        variant.variant_id for variant in variants if variant.assigned.split != "pilot"
    )
    report = {
        "expected_corrupted_pilot_variants": len(expected),
        "decision_count": len(decisions),
        "missing_variant_ids": sorted(set(expected) - observed),
        "unknown_variant_ids": sorted(observed - set(expected)),
        "duplicate_variant_ids": sorted(
            variant_id for variant_id, count in counts.items() if count > 1
        ),
        "non_pilot_variant_ids": non_pilot_variants,
        "answerable_count": sum(decision.answerable for decision in decisions),
        "unanswerable_count": sum(not decision.answerable for decision in decisions),
        "review_methods": sorted({decision.review_method for decision in decisions}),
    }
    report["valid"] = (
        bool(expected)
        and not report["missing_variant_ids"]
        and not report["unknown_variant_ids"]
        and not report["duplicate_variant_ids"]
        and not non_pilot_variants
    )
    return report
