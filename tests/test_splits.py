from dataclasses import replace

from mpvs.data.records import AssignedRecord, NormalizedRecord
from mpvs.splits import SplitSpec, split_records, validate_assigned_records


def _pope_records(image_count=9):
    return [
        NormalizedRecord(
            dataset="pope_coco",
            original_image_id=f"image-{image}",
            image_relpath=f"image-{image}.jpg",
            question_id=f"q-{image}-{question}",
            question="Is there an object?",
            label="yes" if question == 0 else "no",
            source_variant="synthetic",
        )
        for image in range(image_count)
        for question in range(2)
    ]


def _assignment(result):
    return {
        assigned.record.question_id: assigned.split
        for records in result.values()
        for assigned in records
    }


def test_split_is_order_independent_and_keeps_image_questions_together():
    records = _pope_records()
    spec = SplitSpec(seed=17, pilot_fraction=0.2, development_fraction=0.4, test_fraction=0.4)
    forward = split_records(records, spec)
    reverse = split_records(reversed(records), spec)
    assert _assignment(forward) == _assignment(reverse)
    assert all(forward[split] for split in ("pilot", "development", "test"))
    report = validate_assigned_records(item for values in forward.values() for item in values)
    assert report["valid"] is True
    image_splits = {}
    for values in forward.values():
        for assigned in values:
            previous = image_splits.setdefault(assigned.record.original_image_id, assigned.split)
            assert previous == assigned.split


def test_split_group_counts_follow_largest_remainder_targets():
    result = split_records(
        _pope_records(image_count=10),
        SplitSpec(seed=3, pilot_fraction=0.1, development_fraction=0.4, test_fraction=0.5),
    )
    report = validate_assigned_records(item for values in result.values() for item in values)
    assert {split: details["group_count"] for split, details in report["splits"].items()} == {
        "pilot": 1,
        "development": 4,
        "test": 5,
    }


def test_celeba_split_is_identity_disjoint_even_with_multiple_images():
    records = [
        NormalizedRecord(
            dataset="celeba",
            original_image_id=f"{identity}-{image}.jpg",
            image_relpath=f"{identity}-{image}.jpg",
            identity_id=str(identity),
            question_id=f"q-{identity}-{image}",
            question="Is this person wearing eyeglasses?",
            label="yes" if image == 0 else "no",
            attribute="Eyeglasses",
        )
        for identity in range(6)
        for image in range(2)
    ]
    result = split_records(
        records,
        SplitSpec(seed=9, pilot_fraction=0.2, development_fraction=0.4, test_fraction=0.4),
    )
    report = validate_assigned_records(item for values in result.values() for item in values)
    assert report["valid"] is True
    assert report["celeba_identity_leakage"] == []


def test_validator_detects_image_and_identity_leakage():
    record = NormalizedRecord(
        dataset="celeba",
        original_image_id="a.jpg",
        image_relpath="a.jpg",
        identity_id="person-1",
        question_id="q1",
        question="Is this person smiling?",
        label="yes",
        attribute="Smiling",
    )
    conflicting = NormalizedRecord(
        dataset="celeba",
        original_image_id="b.jpg",
        image_relpath="b.jpg",
        identity_id="person-1",
        question_id="q2",
        question="Is this person smiling?",
        label="no",
        attribute="Smiling",
    )
    report = validate_assigned_records(
        [AssignedRecord(record, "pilot"), AssignedRecord(conflicting, "test")]
    )
    assert report["valid"] is False
    assert report["celeba_identity_leakage"] == ["person-1"]
    assert report["group_leakage"] == ["celeba:person-1"]


def test_validator_rejects_duplicate_question_even_within_one_split():
    record = _pope_records(image_count=1)[0]
    report = validate_assigned_records(
        [AssignedRecord(record, "pilot"), AssignedRecord(record, "pilot")]
    )
    assert report["valid"] is False
    assert report["duplicate_question_ids"] == [record.question_id]


def _with_checksum(record, checksum):
    return replace(record, source_image_sha256=checksum)


def test_validator_detects_same_image_bytes_under_two_image_ids():
    first, second = (
        _with_checksum(record, "a" * 64)
        for record in (_pope_records(image_count=2)[0], _pope_records(image_count=2)[2])
    )
    assert first.original_image_id != second.original_image_id
    report = validate_assigned_records(
        [AssignedRecord(first, "development"), AssignedRecord(second, "test")]
    )
    assert report["valid"] is False
    assert report["source_image_checksum_leakage"] == ["a" * 64]
    assert report["original_image_leakage"] == []
    assert report["group_leakage"] == []


def test_split_keeps_duplicate_image_bytes_together_for_every_seed():
    records = [
        _with_checksum(record, f"{record.original_image_id[-1]:0>64}")
        for record in _pope_records(image_count=9)
    ]
    # image-7 is a byte-identical copy of image-2 under a different ID.
    records = [
        _with_checksum(record, "2".rjust(64, "0"))
        if record.original_image_id == "image-7"
        else record
        for record in records
    ]
    spec_values = dict(pilot_fraction=0.2, development_fraction=0.4, test_fraction=0.4)
    for seed in range(25):
        result = split_records(records, SplitSpec(seed=seed, **spec_values))
        assignment = {
            assigned.record.original_image_id: assigned.split
            for rows in result.values()
            for assigned in rows
        }
        assert assignment["image-2"] == assignment["image-7"]
        report = validate_assigned_records(item for rows in result.values() for item in rows)
        assert report["valid"] is True


def test_split_merges_celeba_identities_that_share_image_bytes():
    records = [
        NormalizedRecord(
            dataset="celeba",
            original_image_id=f"{identity}-{image}.jpg",
            image_relpath=f"{identity}-{image}.jpg",
            identity_id=str(identity),
            question_id=f"q-{identity}-{image}",
            question="Is this person smiling?",
            label="yes" if image == 0 else "no",
            attribute="Smiling",
            source_image_sha256=("f" * 64 if (identity, image) in {(1, 0), (4, 1)}
                                 else f"{identity}{image}".rjust(64, "0")),
        )
        for identity in range(6)
        for image in range(2)
    ]
    for seed in range(25):
        result = split_records(
            records,
            SplitSpec(seed=seed, pilot_fraction=0.2, development_fraction=0.4, test_fraction=0.4),
        )
        identity_split = {
            assigned.record.identity_id: assigned.split
            for rows in result.values()
            for assigned in rows
        }
        assert identity_split["1"] == identity_split["4"]


def test_split_without_shared_checksums_matches_plain_group_split():
    plain = _pope_records()
    hashed = [
        _with_checksum(record, record.original_image_id[-1].rjust(64, "0")) for record in plain
    ]
    spec = SplitSpec(seed=17, pilot_fraction=0.2, development_fraction=0.4, test_fraction=0.4)
    assert _assignment(split_records(plain, spec)) == _assignment(split_records(hashed, spec))
