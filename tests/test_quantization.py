import torch
from torch import nn

from mpvs.quantization import (
    allocation_coverage,
    audit_modules,
    classify_component,
    effective_signature,
    nf4_policy,
    requested_candidates,
    unsupported_reason,
)


def test_candidate_grid_is_stable_and_unique():
    candidates = requested_candidates()
    assert len(candidates) == 8
    assert len({candidate.id for candidate in candidates}) == 8
    assert candidates[0].id == "v-fp16_c-fp16_l-fp16"
    assert candidates[-1].id == "v-nf4_c-nf4_l-nf4"


def test_component_classification_uses_verified_boundaries():
    assert classify_component("model.vision_model.encoder.layers.0") == "vision"
    assert classify_component("model.connector.modality_projection") == "connector"
    assert classify_component("model.text_model.layers.0") == "language"
    assert classify_component("lm_head") == "language"
    assert classify_component("model.vision_modelish") is None
    assert classify_component("other") is None


def test_effective_signature_is_deterministic_and_sensitive():
    model = nn.Module()
    model.add_module("lm_head", nn.Linear(2, 2))
    records = audit_modules(model)
    assert effective_signature(records) == effective_signature(records)
    changed = [dict(records[0], quantized_4bit=True)]
    assert effective_signature(records) != effective_signature(changed)


def test_nf4_requires_cuda_and_bitsandbytes():
    candidate = requested_candidates()[1]
    assert unsupported_reason(candidate, "mps")
    assert unsupported_reason(candidate, "cuda", bitsandbytes_available=False)
    assert unsupported_reason(candidate, "cuda", bitsandbytes_available=True) is None
    assert unsupported_reason(requested_candidates()[0], "mps") is None


def test_nf4_policy_skips_only_requested_fp16_component_roots():
    candidate = requested_candidates()[1]  # language NF4 only
    policy = nf4_policy(candidate)
    assert policy["modules_to_not_convert"] == [
        "lm_head",
        "model.vision_model",
        "model.connector",
    ]
    assert policy["quant_type"] == "nf4"
    assert policy["compute_dtype"] == "float16"
    assert policy["double_quant"] is False


def test_allocation_coverage_reports_actual_linear_state():
    model = nn.Module()
    model.model = nn.Module()
    model.model.text_model = nn.Module()
    model.model.text_model.proj = nn.Linear(2, 2, dtype=torch.float16)
    records = audit_modules(model)
    coverage = allocation_coverage(requested_candidates()[0], records)
    assert coverage["components"]["language"]["requested_precision"] == "fp16"
    assert coverage["components"]["language"]["realized_state"] == "fp16"
    assert coverage["components"]["language"]["linear_modules"] == 1
    assert coverage["matches_request"] is True


def test_allocation_coverage_detects_missing_nf4_and_preserves_lm_head():
    records = [
        {
            "name": "model.text_model.layers.0.proj",
            "component": "language",
            "class": "torch.nn.modules.linear.Linear",
            "linear_module": True,
            "quantization_exception": None,
            "quantized_4bit": False,
        },
        {
            "name": "lm_head",
            "component": "language",
            "class": "torch.nn.modules.linear.Linear",
            "linear_module": True,
            "quantization_exception": "output_head_preserved_fp16",
            "quantized_4bit": False,
        },
    ]
    coverage = allocation_coverage(requested_candidates()[1], records)
    assert coverage["matches_request"] is False
    assert coverage["components"]["language"]["mismatched_modules"] == [
        "model.text_model.layers.0.proj"
    ]
    assert coverage["components"]["language"]["preserved_exceptions"] == [
        {"name": "lm_head", "reason": "output_head_preserved_fp16"}
    ]
