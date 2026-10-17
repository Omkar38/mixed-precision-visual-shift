import torch
from torch import nn

from mpvs.memory import measure_model_storage


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float16))
        self.alias = self.weight
        self.register_buffer("scale", torch.ones(2, dtype=torch.float32))


def test_memory_counts_shared_storage_once_and_buffers_separately():
    report = measure_model_storage(SharedModel())
    assert report["parameter_bytes"] == 8
    assert report["persistent_buffer_bytes"] == 8
    assert report["total_weight_storage_bytes"] == 16
    assert report["shared_parameter_references_skipped"] == 1


def test_component_totals_include_unclassified_synthetic_parameters():
    report = measure_model_storage(SharedModel())
    assert report["parameter_bytes_by_component"] == {"unclassified": 8}
