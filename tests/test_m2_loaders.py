import json

import pytest
from PIL import Image

from mpvs.data.celeba import load_celeba_annotations
from mpvs.data.images import attach_source_image_checksums
from mpvs.data.manifests import read_jsonl, write_jsonl_atomic, write_jsonl_immutable
from mpvs.data.pope import load_pope_jsonl


def test_pope_loader_normalizes_explicit_variant_and_fields(tmp_path):
    annotations = tmp_path / "pope.jsonl"
    rows = [
        {"question_id": 2, "image": "coco/a.jpg", "text": "Is there a dog?", "label": "yes"},
        {"question_id": 1, "image": "coco/a.jpg", "text": "Is there a cat?", "label": "no"},
    ]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    records = load_pope_jsonl(annotations, variant="adversarial")
    assert {record.original_image_id for record in records} == {"a"}
    assert {record.group_id for record in records} == {"a"}
    assert {record.source_variant for record in records} == {"adversarial"}
    assert {record.label for record in records} == {"yes", "no"}


def test_celeba_loader_requires_identity_and_explicit_questions(tmp_path):
    attributes = tmp_path / "list_attr_celeba.txt"
    attributes.write_text(
        "2\nSmiling Eyeglasses\n000001.jpg 1 -1\n000002.jpg -1 1\n",
        encoding="utf-8",
    )
    identities = tmp_path / "identity_CelebA.txt"
    identities.write_text("000001.jpg 7\n000002.jpg 8\n", encoding="utf-8")
    records = load_celeba_annotations(
        attributes,
        identities,
        attribute_questions={"Eyeglasses": "Is this person wearing eyeglasses?"},
    )
    assert [record.label for record in records] == ["no", "yes"]
    assert [record.identity_id for record in records] == ["7", "8"]
    assert all(record.attribute == "Eyeglasses" for record in records)


def test_celeba_loader_rejects_unverified_attribute_name(tmp_path):
    attributes = tmp_path / "attrs.txt"
    attributes.write_text("1\nSmiling\n000001.jpg 1\n", encoding="utf-8")
    identities = tmp_path / "ids.txt"
    identities.write_text("000001.jpg 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        load_celeba_annotations(
            attributes,
            identities,
            attribute_questions={"Eyeglasses": "Is this person wearing eyeglasses?"},
        )


def test_image_checksums_and_manifest_bytes_are_deterministic(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_root / "a.png")
    annotations = tmp_path / "pope.jsonl"
    annotations.write_text(
        json.dumps(
            {"question_id": "q1", "image": "a.png", "text": "Is it blue?", "label": "no"}
        )
        + "\n",
        encoding="utf-8",
    )
    records = attach_source_image_checksums(
        load_pope_jsonl(annotations, variant="synthetic"), image_root
    )
    assert len(records[0].source_image_sha256) == 64
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_hash = write_jsonl_atomic(first, [records[0].to_dict()])
    second_hash = write_jsonl_atomic(second, reversed([records[0].to_dict()]))
    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    assert read_jsonl(first)[0]["source_image_sha256"] == records[0].source_image_sha256


def test_immutable_manifest_allows_identical_rerun_but_rejects_replacement(tmp_path):
    output = tmp_path / "manifest.jsonl"
    first_hash = write_jsonl_immutable(output, [{"id": 1}])
    assert write_jsonl_immutable(output, [{"id": 1}]) == first_hash
    with pytest.raises(FileExistsError, match="immutable manifest"):
        write_jsonl_immutable(output, [{"id": 2}])
