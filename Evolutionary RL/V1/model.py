from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQNet(nn.Module):
    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size // 2)
        self.linear2 = nn.Linear(hidden_size // 2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

def add_gaussian_noise_(model: nn.Module, std: float = 0.02) -> None:
    """Initial diversity boost for the genesis population."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * std)

def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}

def uniform_crossover(
    parent_a: Dict[str, torch.Tensor],
    parent_b: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Layer-wise uniform crossover using a 50/50 random binary mask.
    offspring_tensor = torch.where(mask, tensor_A, tensor_B)
    """
    child = {}
    for key in parent_a.keys():
        tensor_a = parent_a[key]
        tensor_b = parent_b[key]

        mask = torch.rand_like(tensor_a, dtype=torch.float32) < 0.5
        child[key] = torch.where(mask, tensor_a, tensor_b)

    return child

def mutate_state_dict(
    state_dict: Dict[str, torch.Tensor],
    mutation_rate: float = 0.10,
    mutation_std: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Apply element-wise Gaussian mutation with Bernoulli mask."""
    mutated = clone_state_dict(state_dict)

    for key, tensor in mutated.items():
        mask = torch.rand_like(tensor, dtype=torch.float32) < mutation_rate
        noise = torch.randn_like(tensor) * mutation_std
        mutated[key] = tensor + (mask * noise)

    return mutated
