import argparse
import json

from PIL import Image

from mpvs.data.cli import (
    create_splits,
    expand_manifest_variants,
    normalize_pope,
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
