"""Command-line interface for M1 candidate listing and model audits."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mpvs.model import ModelSpec, discover_components, load_fp16_reference, resolve_device, run_smoke_inference
from mpvs.quantization import allocation_coverage, audit_modules, candidate_by_id, effective_signature, requested_candidates, unsupported_reason


def _versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "transformers", "accelerate", "Pillow", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _write(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(encoded, end="")


def audit(args: argparse.Namespace) -> int:
    candidate = candidate_by_id(args.candidate)
    spec = ModelSpec(
        model_id=args.model_id,
        model_revision=args.model_revision,
        processor_revision=args.processor_revision,
        device=args.device,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "requested_candidate": candidate.to_dict(),
        "model": spec.to_dict(),
        "environment": {"platform": platform.platform(), "packages": _versions()},
        "load_status": "not_attempted",
        "inference_status": "not_attempted",
        "failure_reason": None,
    }
    try:
        device = resolve_device(spec.device)
        report["resolved_device"] = device
        reason = unsupported_reason(candidate, device)
        if reason:
            report.update(load_status="unsupported", failure_reason=reason)
            _write(report, args.output)
            return 2
        model, processor, device = load_fp16_reference(spec)
        report["load_status"] = "success"
        records = audit_modules(model)
        report["component_modules"] = discover_components(model)
        report["module_audit"] = records
        report["allocation_audit"] = allocation_coverage(candidate, records)
        report["effective_configuration_signature"] = effective_signature(records)
        from mpvs.memory import measure_model_storage

        report["memory"] = measure_model_storage(model)
        if args.smoke_inference:
            report["smoke_inference"] = run_smoke_inference(model, processor, device)
            report["inference_status"] = "success"
        _write(report, args.output)
        return 0
    except Exception as exc:  # preserve a diagnostic report for infrastructure failures
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if report["load_status"] == "not_attempted":
            report["load_status"] = "failed"
        else:
            report["inference_status"] = "failed"
        _write(report, args.output)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mpvs", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("candidates", help="print the stable requested candidate grid")
    audit_parser = subparsers.add_parser("audit", help="audit one freshly loaded candidate")
    audit_parser.add_argument("--candidate", default="v-fp16_c-fp16_l-fp16")
    audit_parser.add_argument("--model-id", default=ModelSpec.model_id)
    audit_parser.add_argument("--model-revision", default=ModelSpec.model_revision)
    audit_parser.add_argument("--processor-revision", default=ModelSpec.processor_revision)
    audit_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    audit_parser.add_argument("--smoke-inference", action="store_true")
    audit_parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "candidates":
        print(json.dumps([candidate.to_dict() for candidate in requested_candidates()], indent=2))
        return 0
    return audit(args)


if __name__ == "__main__":
    sys.exit(main())
