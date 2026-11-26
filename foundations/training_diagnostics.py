import torch
import torch.nn as nn
from typing import List, Dict

class Solution:
    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        l = []
        with torch.no_grad():
            for step in model.children():
                #print(step)
                x = step(x)

                if isinstance(step, nn.Linear):
                    dic = {}
                    dic["mean"] = round(x.mean().item(), 4)
                    dic["std"] = round(x.std().item(), 4)
                    
                    if x.dim() >= 2:
                        dic["dead_fraction"] = round((x <= 0).all(dim=0).float().mean().item(), 4)
                    else:
                        dic["dead_fraction"] = round((x <= 0).float().mean().item(), 4)

                    l.append(dic)
        return l

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        y_pred = model(x)

        criterion = nn.MSELoss()
        loss = criterion(y_pred, y)

        loss.backward()

        l = []
        for step in model.modules():
            if isinstance(step, nn.Linear):
                dic = {}
                gradient = step.weight.grad

                dic["mean"] = round(gradient.mean().item(), 4)
                dic["std"] = round(gradient.std().item(), 4)
                dic["norm"] = round(torch.norm(gradient).item(), 4)

                l.append(dic)
        return l

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for f1 in activation_stats:
            if f1["dead_fraction"] > 0.5:
                return "dead_neurons"

            if f1["std"] < 0.1:
                return "vanishing_gradients"
            elif f1["std"] > 10:
                return "exploding_gradients"

        for f2 in gradient_stats:
            if f2["norm"] > 1000:
                return "exploding_gradients"
            elif f2["norm"] < 1e-5:
                return "vanishing_gradients"
        
        return "healthy"