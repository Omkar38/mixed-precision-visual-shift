import hashlib

import pytest

from mpvs.data.records import NormalizedRecord
from mpvs.data.sampling import (
    CelebASamplingSpec,
    sample_celeba_records,
    validate_celeba_sample,
)


def _checksum(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records(attributes=("Eyeglasses", "Smiling"), identities_per_label=4):
    records = []
    for attribute in attributes:
        question = f"Is this person {attribute.lower()}?"
        for label in ("yes", "no"):
            for index in range(identities_per_label):
                identity = f"{attribute}-{label}-{index}"
                for image in range(2):
                    filename = f"{identity}-{image}.jpg"
                    records.append(
                        NormalizedRecord(
                            dataset="celeba",
                            original_image_id=filename,
                            image_relpath=filename,
                            identity_id=identity,
                            question_id=f"celeba::{filename}::{attribute}",
                            question=question,
                            label=label,
                            attribute=attribute,
                            source_image_sha256=_checksum(filename),
                        )
                    )
    return records


def test_balanced_sample_is_exact_and_respects_identity_and_image_limits():
    spec = CelebASamplingSpec(
        seed=590,
        attributes=("Smiling", "Eyeglasses"),
        per_label_count=2,
    )
    sampled = sample_celeba_records(_records(), spec)
    report = validate_celeba_sample(sampled, spec, require_image_checksums=True)

    assert report["valid"] is True
    assert report["record_count"] == 8
    assert report["attribute_label_counts"] == {
        "Eyeglasses": {"yes": 2, "no": 2},
        "Smiling": {"yes": 2, "no": 2},
    }
    assert report["duplicate_original_image_ids"] == []
    assert report["repeated_identity_attribute_pairs"] == []


def test_sampling_is_independent_of_input_order():
    source = _records()
    spec = CelebASamplingSpec(seed=23, attributes=("Eyeglasses", "Smiling"), per_label_count=2)
    forward = sample_celeba_records(source, spec)
    reverse = sample_celeba_records(reversed(source), spec)
    assert [record.question_id for record in forward] == [
        record.question_id for record in reverse
    ]


def test_sampler_uses_one_question_when_image_has_multiple_attributes():
    shared = []
    for attribute in ("Eyeglasses", "Smiling"):
        shared.append(
            NormalizedRecord(
                dataset="celeba",
                original_image_id="shared.jpg",
                image_relpath="shared.jpg",
                identity_id="shared-person",
                question_id=f"celeba::shared.jpg::{attribute}",
                question=f"Question for {attribute}?",
                label="yes",
                attribute=attribute,
                source_image_sha256=_checksum("shared.jpg"),
            )
        )
    source = _records(identities_per_label=3) + shared
    spec = CelebASamplingSpec(seed=3, attributes=("Eyeglasses", "Smiling"), per_label_count=2)
    sampled = sample_celeba_records(source, spec)
    assert len({record.original_image_id for record in sampled}) == len(sampled)


def test_sampler_fails_clearly_when_requested_count_is_impossible():
    spec = CelebASamplingSpec(seed=1, attributes=("Eyeglasses",), per_label_count=5)
    with pytest.raises(ValueError, match="cannot build requested CelebA sample"):
        sample_celeba_records(_records(attributes=("Eyeglasses",), identities_per_label=2), spec)


def test_final_validation_requires_image_checksums():
    records = [
        NormalizedRecord(
            dataset="celeba",
            original_image_id=f"{label}.jpg",
            image_relpath=f"{label}.jpg",
            identity_id=f"person-{label}",
            question_id=f"q-{label}",
            question="Is this person smiling?",
            label=label,
            attribute="Smiling",
        )
        for label in ("yes", "no")
    ]
    spec = CelebASamplingSpec(seed=1, attributes=("Smiling",), per_label_count=1)
    sampled = sample_celeba_records(records, spec)
    assert validate_celeba_sample(sampled, spec)["valid"] is True
    assert (
        validate_celeba_sample(sampled, spec, require_image_checksums=True)["valid"]
        is False
    )
