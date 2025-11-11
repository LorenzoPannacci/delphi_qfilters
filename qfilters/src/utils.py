import torch
from huggingface_hub import PyTorchModelHubMixin

class QFilters(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_layers, num_kv_heads, kv_head_dim):
        super().__init__()
        self.q_filters = torch.nn.Parameter(torch.randn(num_layers, num_kv_heads, kv_head_dim))