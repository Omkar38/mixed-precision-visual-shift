"""Requested precision candidates and actual quantization-state auditing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

PRECISIONS = ("fp16", "nf4")
COMPONENTS = ("vision", "connector", "language")


@dataclass(frozen=True)
class PrecisionCandidate:
    vision: str
    connector: str
    language: str

    def __post_init__(self) -> None:
        if any(getattr(self, name) not in PRECISIONS for name in COMPONENTS):
            raise ValueError(f"precisions must be one of {PRECISIONS}")

    @property
    def id(self) -> str:
        return (
            f"v-{self.vision}_c-{self.connector}_l-{self.language}"
        )

    @property
    def requires_nf4(self) -> bool:
        return any(getattr(self, name) == "nf4" for name in COMPONENTS)

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, **asdict(self)}


def requested_candidates() -> tuple[PrecisionCandidate, ...]:
    return tuple(PrecisionCandidate(*values) for values in product(PRECISIONS, repeat=3))


def candidate_by_id(candidate_id: str) -> PrecisionCandidate:
    for candidate in requested_candidates():
        if candidate.id == candidate_id:
            return candidate
    raise ValueError(f"unknown candidate: {candidate_id}")


def classify_component(module_path: str) -> str | None:
    mappings = (
        ("model.vision_model", "vision"),
        ("model.connector", "connector"),
        ("model.text_model", "language"),
        ("lm_head", "language"),
    )
    for prefix, component in mappings:
        if module_path == prefix or module_path.startswith(prefix + "."):
            return component
    return None


def module_quantization_state(name: str, module: Any) -> dict[str, Any]:
    parameters = list(module.named_parameters(recurse=False))
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for _, parameter in parameters})
    class_name = f"{type(module).__module__}.{type(module).__qualname__}"
    quantized = (
        "4bit" in class_name.lower()
        or "params4bit" in " ".join(type(p).__name__.lower() for _, p in parameters)
        or hasattr(module, "quant_state")
    )
    quant_type = None
    for _parameter_name, parameter in parameters:
        state = getattr(parameter, "quant_state", None)
        if state is not None:
            quant_type = getattr(state, "quant_type", None) or "present"
            break
    return {
        "name": name,
        "component": classify_component(name),
        "class": class_name,
        "direct_parameter_count": sum(p.numel() for _, p in parameters),
        "stored_dtypes": dtypes,
        "quantized_4bit": quantized,
        "quant_type": quant_type,
    }


def audit_modules(model: Any) -> list[dict[str, Any]]:
    return [
        module_quantization_state(name, module)
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    ]


def allocation_coverage(
    candidate: PrecisionCandidate, records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    records = list(records)
    components: dict[str, Any] = {}
    for component in COMPONENTS:
        component_records = [record for record in records if record["component"] == component]
        quantizable = [
            record for record in component_records if record["class"].endswith(".Linear")
        ]
        quantized = [record for record in quantizable if record["quantized_4bit"]]
        requested = getattr(candidate, component)
        if requested == "fp16":
            realized = "fp16" if not quantized else "mixed_or_unexpected_quantization"
        elif quantizable and len(quantized) == len(quantizable):
            realized = "nf4_supported_linear_layers"
        elif quantized:
            realized = "partial_nf4"
        else:
            realized = "not_nf4"
        components[component] = {
            "requested_precision": requested,
            "realized_state": realized,
            "parameterized_modules": len(component_records),
            "linear_modules": len(quantizable),
            "quantized_4bit_linear_modules": len(quantized),
        }
    return {
        "components": components,
        "unclassified_parameterized_modules": [
            record["name"] for record in records if record["component"] is None
        ],
    }


def effective_signature(records: Iterable[dict[str, Any]]) -> str:
    effective = [
        {
            key: record.get(key)
            for key in ("name", "component", "class", "stored_dtypes", "quantized_4bit", "quant_type")
        }
        for record in records
    ]
    encoded = json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def unsupported_reason(candidate: PrecisionCandidate, device: str) -> str | None:
    if candidate.requires_nf4:
        return (
            "NF4 construction is not implemented in the Mac M1 slice; "
            "it requires a separate bitsandbytes audit on an NVIDIA/CUDA host"
        )
    return None
