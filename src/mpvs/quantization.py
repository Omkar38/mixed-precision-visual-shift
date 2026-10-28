"""Requested precision candidates and actual quantization-state auditing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

PRECISIONS = ("fp16", "nf4")
COMPONENTS = ("vision", "connector", "language")
COMPONENT_ROOTS = {
    "vision": "model.vision_model",
    "connector": "model.connector",
    "language": "model.text_model",
}
PRESERVED_LINEAR_EXCEPTIONS = {"lm_head": "output_head_preserved_fp16"}


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
        (COMPONENT_ROOTS["vision"], "vision"),
        (COMPONENT_ROOTS["connector"], "connector"),
        (COMPONENT_ROOTS["language"], "language"),
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
    double_quant = None
    quant_storage_dtype = None
    for _parameter_name, parameter in parameters:
        state = getattr(parameter, "quant_state", None)
        if state is not None:
            quant_type = getattr(state, "quant_type", None) or "present"
            nested = getattr(state, "state2", None)
            double_quant = nested is not None
            quant_storage_dtype = str(parameter.dtype).removeprefix("torch.")
            break
    source_class = getattr(module, "source_cls", None)
    source_class_name = (
        f"{source_class.__module__}.{source_class.__qualname__}"
        if source_class is not None
        else None
    )
    linear_module = class_name.endswith(".Linear") or "linear4bit" in class_name.lower()
    exception = PRESERVED_LINEAR_EXCEPTIONS.get(name)
    return {
        "name": name,
        "component": classify_component(name),
        "class": class_name,
        "direct_parameter_count": sum(p.numel() for _, p in parameters),
        "stored_dtypes": dtypes,
        "linear_module": linear_module,
        "source_class": source_class_name,
        "quantization_exception": exception,
        "quantized_4bit": quantized,
        "quant_type": quant_type,
        "compute_dtype": (
            str(getattr(module, "compute_dtype")).removeprefix("torch.")
            if getattr(module, "compute_dtype", None) is not None
            else None
        ),
        "double_quant": double_quant,
        "quant_storage_dtype": quant_storage_dtype,
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
            record
            for record in component_records
            if record["linear_module"] and record["quantization_exception"] is None
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
            "mismatched_modules": [
                record["name"]
                for record in quantizable
                if (requested == "nf4") != record["quantized_4bit"]
            ],
            "preserved_exceptions": [
                {"name": record["name"], "reason": record["quantization_exception"]}
                for record in component_records
                if record["quantization_exception"] is not None
            ],
        }
    unclassified_quantized = [
        record["name"]
        for record in records
        if record["component"] is None and record["quantized_4bit"]
    ]
    result = {
        "components": components,
        "unclassified_parameterized_modules": [
            record["name"] for record in records if record["component"] is None
        ],
        "unclassified_quantized_modules": unclassified_quantized,
    }
    result["matches_request"] = not unclassified_quantized and all(
        not details["mismatched_modules"] for details in components.values()
    )
    return result


def effective_signature(records: Iterable[dict[str, Any]]) -> str:
    effective = [
        {
            key: record.get(key)
            for key in (
                "name",
                "component",
                "class",
                "stored_dtypes",
                "quantized_4bit",
                "quant_type",
                "compute_dtype",
                "double_quant",
                "quant_storage_dtype",
            )
        }
        for record in records
    ]
    encoded = json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def unsupported_reason(
    candidate: PrecisionCandidate,
    device: str,
    *,
    bitsandbytes_available: bool | None = None,
) -> str | None:
    if not candidate.requires_nf4:
        return None
    if not device.startswith("cuda"):
        return "NF4 candidates require an NVIDIA/CUDA device"
    if bitsandbytes_available is None:
        bitsandbytes_available = importlib.util.find_spec("bitsandbytes") is not None
    if not bitsandbytes_available:
        return "NF4 candidates require bitsandbytes"
    return None


def nf4_policy(candidate: PrecisionCandidate) -> dict[str, Any]:
    """Return the fixed M1 NF4 policy and component roots kept in FP16."""
    skipped = ["lm_head"] + [
        root
        for component, root in COMPONENT_ROOTS.items()
        if getattr(candidate, component) == "fp16"
    ]
    return {
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "double_quant": False,
        "quant_storage_dtype": "uint8",
        "modules_to_not_convert": skipped,
        "preserved_exceptions": PRESERVED_LINEAR_EXCEPTIONS,
    }


def bitsandbytes_config(candidate: PrecisionCandidate) -> Any:
    """Build the Transformers 4-bit loader config for one requested candidate."""
    if not candidate.requires_nf4:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    policy = nf4_policy(candidate)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=policy["quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=policy["double_quant"],
        bnb_4bit_quant_storage=torch.uint8,
        # Transformers 4.52.4 uses this skip list for both its 8-bit and 4-bit quantizers.
        llm_int8_skip_modules=policy["modules_to_not_convert"],
    )
