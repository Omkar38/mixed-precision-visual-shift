"""CLI wiring for the M2 normalization, split, and variant pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mpvs.corruptions import CorruptionSpec, expand_variants, validate_variant_inheritance
from mpvs.data.celeba import load_celeba_annotations
from mpvs.data.images import attach_source_image_checksums
from mpvs.data.manifests import (
    read_jsonl,
    sha256_file,
    write_json_immutable,
    write_jsonl_immutable,
)
from mpvs.data.pope import load_pope_jsonl
from mpvs.data.records import AssignedRecord, NormalizedRecord, validate_normalized_records
from mpvs.data.review import AnswerabilityDecision, validate_answerability_decisions
from mpvs.splits import SplitSpec, split_records, validate_assigned_records


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _attribute_questions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--attribute-question must use ATTRIBUTE=QUESTION")
        attribute, question = value.split("=", 1)
        attribute, question = attribute.strip(), question.strip()
        if not attribute or not question:
            raise ValueError("--attribute-question must use non-empty ATTRIBUTE=QUESTION")
        if attribute in result:
            raise ValueError(f"duplicate CelebA attribute question: {attribute}")
        result[attribute] = question
    return result


def _write_normalized(
    args: argparse.Namespace,
    records: list[NormalizedRecord],
    provenance: dict[str, Any],
) -> int:
    if not args.skip_image_checksums:
        records = attach_source_image_checksums(records, args.image_root)
    report = validate_normalized_records(
        records, require_image_checksums=not args.skip_image_checksums
    )
    if not report["valid"]:
        raise ValueError(f"normalized records failed validation: {report}")
    digest = write_jsonl_immutable(args.output, (record.to_dict() for record in records))
    metadata = {
        "schema_version": 1,
        "status": "success",
        "manifest": str(args.output),
        "manifest_sha256": digest,
        "image_root": str(args.image_root),
        "image_checksums_required": not args.skip_image_checksums,
        "provenance": provenance,
        "validation": report,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_digest = write_json_immutable(metadata_path, metadata)
    _emit({**metadata, "metadata": str(metadata_path), "metadata_sha256": metadata_digest})
    return 0


def normalize_pope(args: argparse.Namespace) -> int:
    records = load_pope_jsonl(
        args.annotations,
        variant=args.variant,
        image_field=args.image_field,
        question_field=args.question_field,
        label_field=args.label_field,
        question_id_field=args.question_id_field,
        image_id_field=args.image_id_field,
    )
    return _write_normalized(
        args,
        records,
        {
            "dataset": "pope_coco",
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "variant": args.variant,
            "field_mapping": {
                "image": args.image_field,
                "question": args.question_field,
                "label": args.label_field,
                "question_id": args.question_id_field,
                "image_id": args.image_id_field,
            },
        },
    )


def normalize_celeba(args: argparse.Namespace) -> int:
    records = load_celeba_annotations(
        args.attributes,
        args.identities,
        attribute_questions=_attribute_questions(args.attribute_question),
    )
    return _write_normalized(
        args,
        records,
        {
            "dataset": "celeba",
            "attributes": str(args.attributes),
            "attributes_sha256": sha256_file(args.attributes),
            "identities": str(args.identities),
            "identities_sha256": sha256_file(args.identities),
            "attribute_questions": _attribute_questions(args.attribute_question),
        },
    )


def create_splits(args: argparse.Namespace) -> int:
    records = [NormalizedRecord.from_dict(row) for row in read_jsonl(args.input)]
    spec = SplitSpec(
        seed=args.seed,
        pilot_fraction=args.pilot_fraction,
        development_fraction=args.development_fraction,
        test_fraction=args.test_fraction,
    )
    assigned = split_records(records, spec)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {
        split: write_jsonl_immutable(
            args.output_dir / f"{split}.jsonl", (record.to_dict() for record in rows)
        )
        for split, rows in assigned.items()
    }
    validation = validate_assigned_records(
        record for rows in assigned.values() for record in rows
    )
    summary = {
        "schema_version": 1,
        "status": "success" if validation["valid"] else "failed",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "split_spec": spec.to_dict(),
        "manifest_sha256": manifest_hashes,
        "validation": validation,
    }
    summary_path = args.output_dir / "split-summary.json"
    summary_digest = write_json_immutable(summary_path, summary)
    _emit({**summary, "summary_file_sha256": summary_digest})
    return 0 if validation["valid"] else 5


def validate_splits(args: argparse.Namespace) -> int:
    records = [
        AssignedRecord.from_dict(row)
        for input_path in args.inputs
        for row in read_jsonl(input_path)
    ]
    report = validate_assigned_records(records)
    _emit(report)
    return 0 if report["valid"] else 5


def expand_manifest_variants(args: argparse.Namespace) -> int:
    base = [AssignedRecord.from_dict(row) for row in read_jsonl(args.input)]
    raw_specs = json.loads(args.corruptions.read_text(encoding="utf-8"))
    if not isinstance(raw_specs, list):
        raise ValueError("corruption specification must be a JSON array")
    specs = [CorruptionSpec.from_dict(value) for value in raw_specs]
    variants = expand_variants(base, specs)
    validation = validate_variant_inheritance(base, variants)
    if not validation["valid"]:
        raise AssertionError(f"variant inheritance failed: {validation}")
    digest = write_jsonl_immutable(args.output, (variant.to_dict() for variant in variants))
    metadata = {
        "schema_version": 1,
        "status": "success",
        "base_manifest": str(args.input),
        "base_manifest_sha256": sha256_file(args.input),
        "variant_manifest": str(args.output),
        "variant_manifest_sha256": digest,
        "corruption_specs": [spec.to_dict() for spec in specs],
        "validation": validation,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_digest = write_json_immutable(metadata_path, metadata)
    _emit({**metadata, "metadata": str(metadata_path), "metadata_sha256": metadata_digest})
    return 0


def validate_answerability(args: argparse.Namespace) -> int:
    from mpvs.corruptions import VariantRecord

    variants = [VariantRecord.from_dict(row) for row in read_jsonl(args.variants)]
    decisions = [
        AnswerabilityDecision.from_dict(row) for row in read_jsonl(args.decisions)
    ]
    report = validate_answerability_decisions(variants, decisions)
    _emit(report)
    return 0 if report["valid"] else 5


def add_m2_parsers(subparsers: Any) -> None:
    pope = subparsers.add_parser("m2-normalize-pope", help="normalize one POPE JSONL variant")
    pope.add_argument("--annotations", type=Path, required=True)
    pope.add_argument("--image-root", type=Path, required=True)
    pope.add_argument("--variant", required=True)
    pope.add_argument("--output", type=Path, required=True)
    pope.add_argument("--image-field", default="image")
    pope.add_argument("--question-field", default="text")
    pope.add_argument("--label-field", default="label")
    pope.add_argument("--question-id-field", default="question_id")
    pope.add_argument("--image-id-field")
    pope.add_argument("--skip-image-checksums", action="store_true")

    celeba = subparsers.add_parser(
        "m2-normalize-celeba", help="normalize selected CelebA attribute questions"
    )
    celeba.add_argument("--attributes", type=Path, required=True)
    celeba.add_argument("--identities", type=Path, required=True)
    celeba.add_argument("--image-root", type=Path, required=True)
    celeba.add_argument("--attribute-question", action="append", required=True)
    celeba.add_argument("--output", type=Path, required=True)
    celeba.add_argument("--skip-image-checksums", action="store_true")

    split = subparsers.add_parser(
        "m2-split", help="create deterministic group-disjoint split manifests"
    )
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--pilot-fraction", type=float, required=True)
    split.add_argument("--development-fraction", type=float, required=True)
    split.add_argument("--test-fraction", type=float, required=True)

    validate = subparsers.add_parser(
        "m2-validate-splits", help="check image, identity, question, and group leakage"
    )
    validate.add_argument("inputs", nargs="+", type=Path)

    variants = subparsers.add_parser(
        "m2-expand-variants", help="expand a split manifest with explicit corruption specs"
    )
    variants.add_argument("--input", type=Path, required=True)
    variants.add_argument("--corruptions", type=Path, required=True)
    variants.add_argument("--output", type=Path, required=True)

    answerability = subparsers.add_parser(
        "m2-validate-answerability",
        help="validate model-independent decisions for corrupted pilot variants",
    )
    answerability.add_argument("--variants", type=Path, required=True)
    answerability.add_argument("--decisions", type=Path, required=True)


def run_m2(args: argparse.Namespace) -> int:
    handlers = {
        "m2-normalize-pope": normalize_pope,
        "m2-normalize-celeba": normalize_celeba,
        "m2-split": create_splits,
        "m2-validate-splits": validate_splits,
        "m2-expand-variants": expand_manifest_variants,
        "m2-validate-answerability": validate_answerability,
    }
    return handlers[args.command](args)
