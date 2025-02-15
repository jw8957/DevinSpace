import torch
import numpy as np
from typing import Dict, Any, List
import logging

class ModelDebugger:
    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir
        self.logger = logging.getLogger('model_debugger')
        self.logger.setLevel(logging.INFO)
    
    def analyze_gradients(self, model) -> Dict[str, Any]:
        """Analyze gradient statistics for model parameters"""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item(),
                    'zero_grad_percentage': 
                        (param.grad == 0).float().mean().item() * 100
                }
        
        return grad_stats
    
    def analyze_activations(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze activation patterns in model outputs"""
        activation_stats = {}
        
        for layer_name, activations in model_outputs.items():
            if isinstance(activations, torch.Tensor):
                activation_stats[layer_name] = {
                    'mean': activations.mean().item(),
                    'std': activations.std().item(),
                    'dead_neuron_percentage':
                        (activations <= 0).float().mean().item() * 100,
                    'saturation_percentage':
                        (activations >= 0.99).float().mean().item() * 100
                }
        
        return activation_stats
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weight distributions"""
        return {
            'entropy': self._compute_attention_entropy(attention_weights),
            'sparsity': self._compute_attention_sparsity(attention_weights),
            'coverage': self._compute_attention_coverage(attention_weights)
        }
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distributions"""
        # Ensure weights sum to 1
        weights = attention_weights.softmax(dim=-1)
        # Compute entropy
        entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()
        return entropy.item()
    
    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> float:
        """Compute sparsity of attention weights"""
        threshold = 0.1  # Consider weights below this as "sparse"
        sparsity = (attention_weights < threshold).float().mean()
        return sparsity.item()
    
    def _compute_attention_coverage(self, attention_weights: torch.Tensor) -> float:
        """Compute token coverage (how many tokens receive significant attention)"""
        threshold = 0.1  # Significant attention threshold
        coverage = (attention_weights > threshold).float().mean()
        return coverage.item()
    
    def generate_debug_report(self, model, outputs, attention_weights) -> str:
        """Generate comprehensive debug report"""
        grad_stats = self.analyze_gradients(model)
        activation_stats = self.analyze_activations(outputs)
        attention_stats = self.analyze_attention_patterns(attention_weights)
        
        report = "# Model Debug Report\n\n"
        
        # Gradient Analysis
        report += "## Gradient Statistics\n"
        for param_name, stats in grad_stats.items():
            report += f"\n### {param_name}\n"
            for metric, value in stats.items():
                report += f"- {metric}: {value:.4f}\n"
        
        # Activation Analysis
        report += "\n## Activation Statistics\n"
        for layer_name, stats in activation_stats.items():
            report += f"\n### {layer_name}\n"
            for metric, value in stats.items():
                report += f"- {metric}: {value:.4f}\n"
        
        # Attention Analysis
        report += "\n## Attention Statistics\n"
        for metric, value in attention_stats.items():
            report += f"- {metric}: {value:.4f}\n"
        
        return report
