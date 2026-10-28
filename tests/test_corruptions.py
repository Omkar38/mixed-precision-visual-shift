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
