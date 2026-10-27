import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        layers = [m for m in model.modules() if len(list(m.children())) == 0]
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    result.append({
                        'mean': round(x.mean().item(), 4),
                        'std': round(x.std().item(), 4),
                        'dead_fraction': round((x <= 0).all(dim=0).float().mean().item(), 4)
                        # 'dead_fraction': (1.0 - torch.count_nonzero(x) / x.size().numel())
                        # 'dead_fraction': (x == 0).float().mean()
                    })
        return result

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        result = []
        model.zero_grad()
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        loss.backward()
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                result.append({
                    'mean': round(layer.weight.grad.mean().item(), 4),
                    'std': round(layer.weight.grad.std().item(), 4),
                    'norm': round(torch.norm(layer.weight.grad).item(), 4)
                })
        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5:
                return 'dead_neurons'
        for layer in gradient_stats:
            if layer['norm'] > 1000:
                return 'exploding_gradients'
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        for layer in activation_stats:
            if layer['std'] < 0.1:
                return 'vanishing_gradients'
        for layer in activation_stats:
            if layer['std'] > 10.0:
                return 'exploding_gradients'
        return 'healthy'