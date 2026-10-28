import argparse
import json

from mpvs.cli import summarize
from mpvs.quantization import requested_candidates


def _report(candidate_id, signature, weight_bytes):
    return {
        "requested_candidate": {"id": candidate_id},
        "configuration_status": "supported",
        "load_status": "success",
        "inference_status": "success",
        "failure_reason": None,
        "effective_configuration_signature": signature,
        "memory": {"total_weight_storage_bytes": weight_bytes},
    }


def test_summary_detects_distinct_candidates_and_common_cap(tmp_path):
    inputs = []
    reports = [
        _report(candidate.id, f"sig-{index}", 10 + index)
        for index, candidate in enumerate(requested_candidates())
    ]
    for index, report in enumerate(reports):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "summary.json"
    assert summarize(argparse.Namespace(inputs=inputs, output=output)) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["m1_candidate_acceptance_met"] is True
    assert result["grid_complete"] is True
    assert result["candidate_caps_with_at_least_two_distinct_configs"][0] == {
        "weight_cap_bytes": 11,
        "eligible_candidate_ids": [reports[0]["requested_candidate"]["id"], reports[1]["requested_candidate"]["id"]],
    }


def test_summary_deduplicates_effective_signatures(tmp_path):
    inputs = []
    reports = [
        _report(candidate.id, "same", 10)
        for candidate in requested_candidates()
    ]
    for index, report in enumerate(reports):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "summary.json"
    assert summarize(argparse.Namespace(inputs=inputs, output=output)) == 4
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["distinct_successful_count"] == 1
    assert result["m1_candidate_acceptance_met"] is False
