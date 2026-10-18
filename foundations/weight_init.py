import torch
import torch.nn as nn
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        torch.manual_seed(0)

        std = (2/(fan_in+fan_out))**(1/2)
        w = (torch.randn(fan_out, fan_in)) * std
        w = (w * 10000) / 10000

        return w.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        torch.manual_seed(0)

        std = (2/fan_in)**(1/2)
        w = (torch.randn(fan_out, fan_in)) * std
        w = (w * 10000) / 10000

        return w.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        l = []

        for i in range(num_layers):
            if i == 0:
                fan_in = input_dim
            else:
                fan_in = hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2.0 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / fan_in)
            else:
                std = 1.0

            w = torch.randn(fan_out, fan_in) * std
            l.append(w)

        x = torch.randn(input_dim) #features du forward pass
        activations = []

        for w in l:
            x = w @ x
            x = torch.relu(x)
            activations.append(round(x.std().item(), 2))

        return activations