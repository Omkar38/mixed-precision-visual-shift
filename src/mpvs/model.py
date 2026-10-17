"""SmolVLM model loading, component discovery, and smoke inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    processor_revision: str = MODEL_REVISION
    device: str = "auto"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_fp16_reference(spec: ModelSpec) -> tuple[Any, Any, str]:
    """Load a fresh FP16 reference directly from its immutable checkpoint."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = resolve_device(spec.device)
    processor = AutoProcessor.from_pretrained(
        spec.model_id, revision=spec.processor_revision
    )
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, processor, device


def discover_components(model: Any) -> dict[str, list[str]]:
    """Inventory module paths using the verified SmolVLM component mapping."""
    from mpvs.quantization import classify_component

    found: dict[str, list[str]] = {
        "vision": [],
        "connector": [],
        "language": [],
        "unclassified": [],
    }
    for name, _module in model.named_modules():
        if not name:
            continue
        found[classify_component(name) or "unclassified"].append(name)
    return found


def run_smoke_inference(model: Any, processor: Any, device: str) -> dict[str, Any]:
    """Run one deterministic synthetic image/question generation."""
    from PIL import Image

    image = Image.new("RGB", (384, 384), color=(32, 96, 160))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Is the image blue? Answer yes or no."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_dtype = next(model.parameters()).dtype
    inputs = {
        key: value.to(device=device, dtype=model_dtype) if value.is_floating_point() else value.to(device)
        for key, value in inputs.items()
    }
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=4)
    prompt_length = inputs["input_ids"].shape[-1]
    answer = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    return {
        "status": "success",
        "synthetic_input": True,
        "answer": answer,
        "max_new_tokens": 4,
        "do_sample": False,
    }
