from mpvs.corruptions import CorruptionSpec, expand_variants
from mpvs.data.records import AssignedRecord, NormalizedRecord
from mpvs.data.review import AnswerabilityDecision, validate_answerability_decisions


def _variants():
    record = AssignedRecord(
        NormalizedRecord(
            dataset="pope_coco",
            original_image_id="image-1",
            image_relpath="image-1.jpg",
            question_id="q1",
            question="Is there a dog?",
            label="yes",
            source_variant="synthetic",
        ),
        "pilot",
    )
    return expand_variants(
        [record],
        [
            CorruptionSpec("clean", "clean"),
            CorruptionSpec("darkness", "dark-1", 0.5),
            CorruptionSpec("gaussian_blur", "blur-1", 1.0),
        ],
    )


def test_answerability_requires_one_decision_per_corrupted_pilot_variant():
    variants = _variants()
    decisions = [
        AnswerabilityDecision(
            variant.variant_id,
            answerable=variant.corruption.kind == "darkness",
            review_method="human_model_blind_review",
            reason=None if variant.corruption.kind == "darkness" else "object is no longer visible",
        )
        for variant in variants
        if variant.corruption.kind != "clean"
    ]
    report = validate_answerability_decisions(variants, decisions)
    assert report["valid"] is True
    assert report["answerable_count"] == 1
    assert report["unanswerable_count"] == 1


def test_answerability_detects_missing_decision():
    variants = _variants()
    report = validate_answerability_decisions(variants, [])
    assert report["valid"] is False
    assert len(report["missing_variant_ids"]) == 2
