import argparse
import json

from mpvs.cli import audit


def test_unsupported_audit_preserves_candidate_identity(tmp_path):
    output = tmp_path / "unsupported.json"
    args = argparse.Namespace(
        candidate="v-fp16_c-fp16_l-nf4",
        model_id="unused",
        model_revision="unused",
        processor_revision="unused",
        device="cpu",
        smoke_inference=True,
        output=output,
    )

    assert audit(args) == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["candidate_id"] == args.candidate
    assert report["requested_candidate"]["id"] == args.candidate
    assert report["supported"] is False
    assert report["load_status"] == "unsupported"
    assert report["inference_status"] == "not_attempted"
