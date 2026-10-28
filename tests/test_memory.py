import torch
from torch import nn

from mpvs.memory import measure_model_storage


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float16))
        self.alias = self.weight
        self.register_buffer("scale", torch.ones(2, dtype=torch.float32))


class FakeQuantState:
    def __init__(self):
        self.absmax = torch.ones(3, dtype=torch.float32)
        self.code = torch.ones(4, dtype=torch.uint8)
        self.state2 = None


class QuantizedLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.uint8), requires_grad=False)
        self.weight.quant_state = FakeQuantState()


class ParameterBufferAliasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float16))
        self.register_buffer("weight_view", self.weight.detach())


def test_memory_counts_shared_storage_once_and_buffers_separately():
    report = measure_model_storage(SharedModel())
    assert report["parameter_bytes"] == 8
    assert report["persistent_buffer_bytes"] == 8
    assert report["total_weight_storage_bytes"] == 16
    assert report["shared_parameter_references_skipped"] == 1


def test_component_totals_include_unclassified_synthetic_parameters():
    report = measure_model_storage(SharedModel())
    assert report["parameter_bytes_by_component"] == {"unclassified": 8}


def test_memory_counts_quantization_state_tensor_storage():
    report = measure_model_storage(QuantizedLikeModel())
    assert report["parameter_bytes"] == 4
    assert report["quantization_metadata_bytes"] == 16
    assert report["total_weight_storage_bytes"] == 20
    assert {item["name"] for item in report["quantization_metadata_tensors"]} == {
        "weight.quant_state.absmax",
        "weight.quant_state.code",
    }


def test_memory_does_not_recount_parameter_storage_exposed_as_buffer():
    report = measure_model_storage(ParameterBufferAliasModel())
    assert report["parameter_bytes"] == 8
    assert report["persistent_buffer_bytes"] == 0
    assert report["shared_buffer_references_skipped"] == 1
    assert report["total_weight_storage_bytes"] == 8
