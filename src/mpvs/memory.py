"""Unique model storage accounting without double-counting shared tensors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _storage_identity(tensor: Any) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    pointer = storage.data_ptr()
    if pointer:
        return (str(tensor.device), pointer, storage.nbytes())
    return (str(tensor.device), "object", id(storage), storage.nbytes())


def _unique_storage_bytes(named_tensors: Iterable[tuple[str, Any]]) -> tuple[int, int]:
    seen: set[tuple[Any, ...]] = set()
    total = 0
    duplicates = 0
    for _name, tensor in named_tensors:
        identity = _storage_identity(tensor)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        total += tensor.untyped_storage().nbytes()
    return total, duplicates


def measure_model_storage(model: Any) -> dict[str, Any]:
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    parameter_bytes, shared_parameters = _unique_storage_bytes(parameters)
    buffer_bytes, shared_buffers = _unique_storage_bytes(buffers)

    by_component: dict[str, int] = defaultdict(int)
    seen: set[tuple[Any, ...]] = set()
    from mpvs.quantization import classify_component

    for name, tensor in parameters:
        identity = _storage_identity(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        by_component[classify_component(name) or "unclassified"] += tensor.untyped_storage().nbytes()

    return {
        "counting_convention": "unique underlying parameter and persistent-buffer storages",
        "parameter_bytes": parameter_bytes,
        "persistent_buffer_bytes": buffer_bytes,
        "quantization_metadata_bytes": 0,
        "total_weight_storage_bytes": parameter_bytes + buffer_bytes,
        "shared_parameter_references_skipped": shared_parameters,
        "shared_buffer_references_skipped": shared_buffers,
        "parameter_bytes_by_component": dict(sorted(by_component.items())),
        "metadata_note": "NF4 metadata accounting remains pending CUDA/bitsandbytes validation",
    }
