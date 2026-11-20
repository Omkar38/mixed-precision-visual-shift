import argparse
import json

from PIL import Image

from mpvs.data.cli import (
    create_splits,
    expand_manifest_variants,
    normalize_pope,
    sample_celeba,
    verify_manifest_images,
    validate_splits,
)
from mpvs.data.manifests import read_jsonl, write_jsonl_immutable
from mpvs.data.records import NormalizedRecord


def test_m2_normalize_pope_writes_provenance_sidecar(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image_root / "image.png")
    annotations = tmp_path / "pope.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "image": "image.png",
                "text": "Is an object present?",
                "label": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"
    assert normalize_pope(
        argparse.Namespace(
            annotations=annotations,
            image_root=image_root,
            variant="synthetic",
            output=output,
            image_field="image",
            question_field="text",
            label_field="label",
            question_id_field="question_id",
            image_id_field=None,
            skip_image_checksums=False,
        )
    ) == 0
    metadata = json.loads(
        output.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["provenance"]["variant"] == "synthetic"
    assert len(metadata["provenance"]["annotations_sha256"]) == 64
    assert len(metadata["manifest_sha256"]) == 64


def test_m2_split_validate_and_expand_cli_handlers(tmp_path):
    records = [
        NormalizedRecord(
            dataset="pope_coco",
            original_image_id=f"image-{index}",
            image_relpath=f"image-{index}.jpg",
            question_id=f"q-{index}",
            question="Is an object present?",
            label="yes" if index % 2 else "no",
            source_variant="synthetic",
        )
        for index in range(9)
    ]
    normalized = tmp_path / "normalized.jsonl"
    write_jsonl_immutable(normalized, (record.to_dict() for record in records))
    output_dir = tmp_path / "splits"
    assert create_splits(
        argparse.Namespace(
            input=normalized,
            output_dir=output_dir,
            seed=5,
            pilot_fraction=0.2,
            development_fraction=0.4,
            test_fraction=0.4,
        )
    ) == 0
    manifests = [output_dir / f"{split}.jsonl" for split in ("pilot", "development", "test")]
    assert validate_splits(argparse.Namespace(inputs=manifests)) == 0
    corruption_path = tmp_path / "corruptions.json"
    corruption_path.write_text(
        json.dumps(
            [
                {"kind": "clean", "severity_id": "clean", "value": None},
                {"kind": "darkness", "severity_id": "synthetic-dark", "value": 0.5},
            ]
        ),
        encoding="utf-8",
    )
    variants = tmp_path / "pilot-variants.jsonl"
    assert expand_manifest_variants(
        argparse.Namespace(
            input=output_dir / "pilot.jsonl",
            corruptions=corruption_path,
            output=variants,
        )
    ) == 0
    assert len(read_jsonl(variants)) == 2 * len(read_jsonl(output_dir / "pilot.jsonl"))


def test_m2_sample_celeba_writes_balanced_immutable_manifest(tmp_path):
    records = [
        NormalizedRecord(
            dataset="celeba",
            original_image_id=f"{label}-{index}.jpg",
            image_relpath=f"{label}-{index}.jpg",
            identity_id=f"{label}-{index}",
            question_id=f"q-{label}-{index}",
            question="Is this person smiling?",
            label=label,
            attribute="Smiling",
        )
        for label in ("yes", "no")
        for index in range(3)
    ]
    source = tmp_path / "celeba-normalized.jsonl"
    write_jsonl_immutable(source, (record.to_dict() for record in records))
    output = tmp_path / "celeba-sampled.jsonl"
    args = argparse.Namespace(
        input=source,
        attribute=["Smiling"],
        per_label=2,
        seed=590,
        output=output,
        allow_missing_image_checksums=True,
    )
    assert sample_celeba(args) == 0
    assert sample_celeba(args) == 0
    sampled = read_jsonl(output)
    assert len(sampled) == 4
    assert {label: sum(row["label"] == label for row in sampled) for label in ("yes", "no")} == {
        "yes": 2,
        "no": 2,
    }
    metadata = json.loads(
        output.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["sampling_spec"]["seed"] == 590
    assert metadata["validation"]["valid"] is True


def test_m2_verify_images_attaches_checksums_to_a_sample(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_root / "selected.jpg")
    source = tmp_path / "sample-unverified.jsonl"
    write_jsonl_immutable(
        source,
        [
            NormalizedRecord(
                dataset="celeba",
                original_image_id="selected.jpg",
                image_relpath="selected.jpg",
                identity_id="person-1",
                question_id="q-selected",
                question="Is this person smiling?",
                label="yes",
                attribute="Smiling",
            ).to_dict()
        ],
    )
    output = tmp_path / "sample-verified.jsonl"
    assert verify_manifest_images(
        argparse.Namespace(input=source, image_root=image_root, output=output)
    ) == 0
    row = read_jsonl(output)[0]
    assert len(row["source_image_sha256"]) == 64
    metadata = json.loads(
        output.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["validation"]["missing_source_image_checksums"] == []
