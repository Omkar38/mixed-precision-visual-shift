"""Command-line interface for the MPVS model audit and data pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mpvs.model import ModelSpec, discover_components, load_candidate, resolve_device, run_smoke_inference
from mpvs.quantization import allocation_coverage, audit_modules, candidate_by_id, effective_signature, requested_candidates, unsupported_reason


def _versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "Pillow", "numpy"):
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
        "candidate_id": candidate.id,
        "requested_candidate": candidate.to_dict(),
        "model": spec.to_dict(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": _versions(),
        },
        "load_status": "not_attempted",
        "inference_status": "not_attempted",
        "configuration_status": "not_audited",
        "supported": None,
        "failure_stage": None,
        "failure_reason": None,
    }
    stage = "device_check"
    try:
        device = resolve_device(spec.device)
        report["resolved_device"] = device
        reason = unsupported_reason(candidate, device)
        if reason:
            report.update(
                load_status="unsupported",
                supported=False,
                failure_stage="compatibility_check",
                failure_reason=reason,
            )
            _write(report, args.output)
            return 2
        if device.startswith("cuda"):
            import torch

            report["environment"]["cuda"] = {
                "runtime_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
            }
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        stage = "load"
        load_started = time.perf_counter()
        model, processor, device, policy = load_candidate(spec, candidate)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        report["load_status"] = "success"
        report["load_seconds"] = time.perf_counter() - load_started
        report["load_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        )
        report["load_peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved() if device.startswith("cuda") else None
        )
        report["quantization_policy"] = policy
        stage = "configuration_audit"
        records = audit_modules(model)
        report["component_modules"] = discover_components(model)
        report["module_audit"] = records
        coverage = allocation_coverage(candidate, records)
        report["allocation_audit"] = coverage
        report["effective_configuration_signature"] = effective_signature(records)
        from mpvs.memory import measure_model_storage

        report["memory"] = measure_model_storage(model)
        report["configuration_status"] = (
            "supported" if coverage["matches_request"] else "realization_mismatch"
        )
        report["supported"] = coverage["matches_request"]
        if not coverage["matches_request"]:
            report["failure_reason"] = "Requested precision allocation did not match audited modules"
            _write(report, args.output)
            return 3
        if args.smoke_inference:
            stage = "inference"
            report["smoke_inference"] = run_smoke_inference(model, processor, device)
            report["inference_status"] = "success"
        _write(report, args.output)
        return 0
    except Exception as exc:  # preserve a diagnostic report for infrastructure failures
        report["failure_stage"] = stage
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        if stage in {"device_check", "load"}:
            report["load_status"] = "failed"
        elif stage == "inference":
            report["inference_status"] = "failed"
        else:
            report["configuration_status"] = "audit_failed"
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
    summary_parser = subparsers.add_parser(
        "summarize", help="summarize isolated candidate audit JSON files"
    )
    summary_parser.add_argument("inputs", nargs="+", type=Path)
    summary_parser.add_argument("--output", type=Path)
    from mpvs.data.cli import add_m2_parsers

    add_m2_parsers(subparsers)
    return parser


def summarize(args: argparse.Namespace) -> int:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    expected_ids = {candidate.id for candidate in requested_candidates()}
    observed_ids = {report["requested_candidate"]["id"] for report in reports}
    successful = [
        report
        for report in reports
        if report.get("configuration_status") == "supported"
        and report.get("load_status") == "success"
        and report.get("inference_status") == "success"
    ]
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for report in successful:
        by_signature.setdefault(report["effective_configuration_signature"], []).append(report)
    distinct = [group[0] for group in by_signature.values()]
    distinct.sort(key=lambda report: report["memory"]["total_weight_storage_bytes"])
    caps = []
    for cap in sorted(
        {report["memory"]["total_weight_storage_bytes"] for report in distinct}
    ):
        eligible = [
            report["requested_candidate"]["id"]
            for report in distinct
            if report["memory"]["total_weight_storage_bytes"] <= cap
        ]
        if len(eligible) >= 2:
            caps.append({"weight_cap_bytes": cap, "eligible_candidate_ids": eligible})
    summary = {
        "schema_version": 1,
        "input_count": len(reports),
        "grid_complete": observed_ids == expected_ids,
        "missing_candidate_ids": sorted(expected_ids - observed_ids),
        "unexpected_candidate_ids": sorted(observed_ids - expected_ids),
        "successful_candidate_ids": [
            report["requested_candidate"]["id"] for report in successful
        ],
        "failed_or_unsupported": [
            {
                "candidate_id": report["requested_candidate"]["id"],
                "load_status": report.get("load_status"),
                "inference_status": report.get("inference_status"),
                "configuration_status": report.get("configuration_status"),
                "failure_reason": report.get("failure_reason"),
            }
            for report in reports
            if report not in successful
        ],
        "effective_configuration_groups": [
            {
                "signature": signature,
                "candidate_ids": [item["requested_candidate"]["id"] for item in group],
            }
            for signature, group in sorted(by_signature.items())
        ],
        "distinct_successful_count": len(distinct),
        "candidate_weight_bytes": {
            report["requested_candidate"]["id"]: report["memory"]["total_weight_storage_bytes"]
            for report in distinct
        },
        "candidate_caps_with_at_least_two_distinct_configs": caps,
        "m1_candidate_acceptance_met": (
            observed_ids == expected_ids and len(distinct) >= 2 and bool(caps)
        ),
    }
    _write(summary, args.output)
    return 0 if summary["m1_candidate_acceptance_met"] else 4


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "candidates":
        print(json.dumps([candidate.to_dict() for candidate in requested_candidates()], indent=2))
        return 0
    if args.command == "summarize":
        return summarize(args)
    if args.command.startswith("m2-"):
        from mpvs.data.cli import run_m2

        return run_m2(args)
    return audit(args)


if __name__ == "__main__":
    sys.exit(main())
