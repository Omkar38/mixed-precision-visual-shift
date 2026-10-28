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


def _unique_storage_bytes(
    named_tensors: Iterable[tuple[str, Any]],
    seen: set[tuple[Any, ...]] | None = None,
) -> tuple[int, int, set[tuple[Any, ...]]]:
    seen = set() if seen is None else seen
    total = 0
    duplicates = 0
    for _name, tensor in named_tensors:
        identity = _storage_identity(tensor)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        total += tensor.untyped_storage().nbytes()
    return total, duplicates, seen


def _nested_tensors(value: Any, prefix: str, seen: set[int]) -> Iterable[tuple[str, Any]]:
    """Yield tensors from a bitsandbytes quantization-state object."""
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _nested_tensors(child, f"{prefix}.{key}", seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _nested_tensors(child, f"{prefix}.{index}", seen)
    elif hasattr(value, "__dict__"):
        for key, child in vars(value).items():
            yield from _nested_tensors(child, f"{prefix}.{key}", seen)


def quantization_metadata_tensors(model: Any) -> list[tuple[str, Any]]:
    tensors: list[tuple[str, Any]] = []
    seen_objects: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        state = getattr(parameter, "quant_state", None)
        if state is not None:
            tensors.extend(_nested_tensors(state, f"{name}.quant_state", seen_objects))
    return tensors


def measure_model_storage(model: Any) -> dict[str, Any]:
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    parameter_bytes, shared_parameters, occupied = _unique_storage_bytes(parameters)
    buffer_bytes, shared_buffers, occupied = _unique_storage_bytes(buffers, occupied)
    metadata_tensors = quantization_metadata_tensors(model)
    metadata_seen: set[tuple[Any, ...]] = set()
    metadata_bytes = 0
    metadata_entries: list[dict[str, Any]] = []
    for name, tensor in metadata_tensors:
        identity = _storage_identity(tensor)
        if identity in occupied or identity in metadata_seen:
            continue
        metadata_seen.add(identity)
        storage_bytes = tensor.untyped_storage().nbytes()
        metadata_bytes += storage_bytes
        metadata_entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "storage_bytes": storage_bytes,
            }
        )

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
        "counting_convention": "unique underlying storage across parameters, persistent buffers, and exposed quantization metadata",
        "parameter_bytes": parameter_bytes,
        "persistent_buffer_bytes": buffer_bytes,
        "quantization_metadata_bytes": metadata_bytes,
        "total_weight_storage_bytes": parameter_bytes + buffer_bytes + metadata_bytes,
        "shared_parameter_references_skipped": shared_parameters,
        "shared_buffer_references_skipped": shared_buffers,
        "parameter_bytes_by_component": dict(sorted(by_component.items())),
        "quantization_metadata_tensors": metadata_entries,
        "metadata_note": "Counts unique tensor storages reachable from parameter quant_state objects",
    }
