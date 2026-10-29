import pytest
from PIL import Image

from mpvs.corruptions import (
    CorruptionSpec,
    apply_corruption,
    expand_variants,
    validate_variant_inheritance,
)
from mpvs.data.records import AssignedRecord, NormalizedRecord


def _assigned():
    return AssignedRecord(
        NormalizedRecord(
            dataset="pope_coco",
            original_image_id="image-1",
            image_relpath="image-1.png",
            question_id="q1",
            question="Is the image visible?",
            label="yes",
            source_variant="synthetic",
        ),
        "pilot",
    )


def test_darkness_and_blur_are_deterministic():
    image = Image.new("RGB", (8, 8), (100, 120, 140))
    darkness = CorruptionSpec("darkness", "dark-1", 0.5)
    blur = CorruptionSpec("gaussian_blur", "blur-1", 1.25)
    assert apply_corruption(image, darkness).tobytes() == apply_corruption(image, darkness).tobytes()
    assert apply_corruption(image, blur).tobytes() == apply_corruption(image, blur).tobytes()
    assert apply_corruption(image, darkness).getpixel((0, 0)) == (50, 60, 70)


def test_variants_inherit_question_group_and_split():
    base = [_assigned()]
    specs = [
        CorruptionSpec("clean", "clean"),
        CorruptionSpec("darkness", "dark-1", 0.5),
        CorruptionSpec("gaussian_blur", "blur-1", 1.0),
    ]
    variants = expand_variants(base, specs)
    assert len(variants) == 3
    assert {variant.assigned.split for variant in variants} == {"pilot"}
    assert {variant.assigned.record.group_id for variant in variants} == {"image-1"}
    assert all(item.to_dict()["transform_stage"] == "before_model_processor" for item in variants)
    assert validate_variant_inheritance(base, variants)["valid"] is True


def test_corruption_signatures_change_with_parameters():
    first = CorruptionSpec("darkness", "dark-1", 0.5)
    second = CorruptionSpec("darkness", "dark-2", 0.25)
    assert first.signature != second.signature


def _paired_base():
    return [
        AssignedRecord(
            NormalizedRecord(
                dataset="pope_coco",
                original_image_id=f"image-{index}",
                image_relpath=f"image-{index}.png",
                question_id=f"q{index}",
                question="Is the image visible?",
                label="yes",
                source_variant="synthetic",
            ),
            "development",
        )
        for index in range(2)
    ]


def _paired_specs():
    return [
        CorruptionSpec("clean", "clean"),
        CorruptionSpec("darkness", "dark-1", 0.5),
        CorruptionSpec("gaussian_blur", "blur-1", 1.0),
    ]


def test_inheritance_rejects_a_question_missing_one_paired_variant():
    base = _paired_base()
    specs = _paired_specs()
    variants = [
        variant
        for variant in expand_variants(base, specs)
        if variant.variant_id != "q1::darkness::dark-1"
    ]
    for declared in (None, specs):
        report = validate_variant_inheritance(base, variants, declared)
        assert report["valid"] is False
        assert report["missing_variants"] == ["q1::darkness::dark-1"]


def test_inheritance_rejects_a_condition_missing_for_every_question_when_specs_given():
    base = _paired_base()
    specs = _paired_specs()
    variants = [
        variant for variant in expand_variants(base, specs) if variant.corruption.kind != "darkness"
    ]
    # Without the declared specs the dropped condition is invisible.
    assert validate_variant_inheritance(base, variants)["valid"] is True
    report = validate_variant_inheritance(base, variants, specs)
    assert report["valid"] is False
    assert report["missing_variants"] == ["q0::darkness::dark-1", "q1::darkness::dark-1"]


def test_inheritance_rejects_a_question_with_no_variants():
    base = _paired_base()
    variants = [
        variant
        for variant in expand_variants(base, _paired_specs())
        if variant.assigned.record.question_id != "q1"
    ]
    report = validate_variant_inheritance(base, variants)
    assert report["valid"] is False
    assert report["missing_variants"] == [
        "q1::clean::clean",
        "q1::darkness::dark-1",
        "q1::gaussian_blur::blur-1",
    ]


def test_inheritance_rejects_one_severity_id_with_two_values():
    base = _paired_base()
    variants = list(expand_variants(base[:1], _paired_specs())) + list(
        expand_variants(
            base[1:],
            [
                CorruptionSpec("clean", "clean"),
                CorruptionSpec("darkness", "dark-1", 0.25),
                CorruptionSpec("gaussian_blur", "blur-1", 1.0),
            ],
        )
    )
    report = validate_variant_inheritance(base, variants)
    assert report["valid"] is False
    assert report["inconsistent_corruption_specs"] == ["darkness::dark-1"]
    assert report["missing_variants"] == []


def test_inheritance_rejects_variants_outside_the_declared_specs():
    base = _paired_base()
    specs = _paired_specs()
    variants = expand_variants(base, specs)
    report = validate_variant_inheritance(base, variants, specs[:2])
    assert report["valid"] is False
    assert report["undeclared_corruption_specs"] == ["gaussian_blur::blur-1"]


def test_numeric_spelling_does_not_change_signature():
    """2 and 2.0 are the same transform, so they must share one cache signature."""
    assert (
        CorruptionSpec(kind="gaussian_blur", severity_id="s1", value=2).signature
        == CorruptionSpec(kind="gaussian_blur", severity_id="s1", value=2.0).signature
    )


def test_boolean_value_is_rejected():
    """bool is an int subclass; True would otherwise pass as radius 1.0."""
    with pytest.raises(ValueError):
        CorruptionSpec(kind="gaussian_blur", severity_id="s1", value=True)
