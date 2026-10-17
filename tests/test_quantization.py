import torch
from torch import nn

from mpvs.quantization import (
    allocation_coverage,
    audit_modules,
    classify_component,
    effective_signature,
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


def test_nf4_is_explicitly_unsupported_off_cuda():
    candidate = requested_candidates()[1]
    assert unsupported_reason(candidate, "mps")
    assert unsupported_reason(candidate, "cuda")
    assert unsupported_reason(requested_candidates()[0], "mps") is None


def test_allocation_coverage_reports_actual_linear_state():
    model = nn.Module()
    model.add_module("lm_head", nn.Linear(2, 2, dtype=torch.float16))
    records = audit_modules(model)
    coverage = allocation_coverage(requested_candidates()[0], records)
    assert coverage["components"]["language"]["requested_precision"] == "fp16"
    assert coverage["components"]["language"]["realized_state"] == "fp16"
    assert coverage["components"]["language"]["linear_modules"] == 1
