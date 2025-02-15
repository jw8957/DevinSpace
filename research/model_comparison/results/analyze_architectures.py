import torch
import numpy as np
from collections import defaultdict

def count_model_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_architecture(model):
    """Analyze model architecture details"""
    architecture_info = {
        'total_params': count_model_parameters(model),
        'layers': defaultdict(int),
        'activation_functions': defaultdict(int)
    }
    
    # Analyze layer types
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
            architecture_info['layers']['lstm'] += 1
        elif isinstance(module, torch.nn.MultiheadAttention):
            architecture_info['layers']['attention'] += 1
        elif isinstance(module, torch.nn.Linear):
            architecture_info['layers']['linear'] += 1
        
        # Count activation functions
        if isinstance(module, torch.nn.ReLU):
            architecture_info['activation_functions']['relu'] += 1
        elif isinstance(module, torch.nn.Tanh):
            architecture_info['activation_functions']['tanh'] += 1
        elif isinstance(module, torch.nn.Sigmoid):
            architecture_info['activation_functions']['sigmoid'] += 1
    
    return architecture_info

def format_architecture_summary(model_name, arch_info):
    """Format architecture information into readable summary"""
    summary = f"## {model_name} Architecture\n\n"
    summary += f"### Parameters\n"
    summary += f"- Total trainable parameters: {arch_info['total_params']:,}\n\n"
    
    summary += "### Layer Composition\n"
    for layer_type, count in arch_info['layers'].items():
        summary += f"- {layer_type.upper()} layers: {count}\n"
    summary += "\n"
    
    summary += "### Activation Functions\n"
    for func_type, count in arch_info['activation_functions'].items():
        summary += f"- {func_type.upper()}: {count}\n"
    
    return summary

def compare_architectures(bilstm_model, attn_model):
    """Compare model architectures and generate report"""
    bilstm_info = analyze_model_architecture(bilstm_model)
    attn_info = analyze_model_architecture(attn_model)
    
    report = "# Model Architecture Comparison\n\n"
    report += format_architecture_summary("BiLSTM+Attention", bilstm_info)
    report += "\n" + format_architecture_summary("Attention-only", attn_info)
    
    # Add comparison summary
    report += "\n## Architecture Comparison Summary\n"
    report += f"- Parameter Ratio: {bilstm_info['total_params'] / attn_info['total_params']:.2f}x\n"
    report += "- Key Differences:\n"
    report += "  - Sequence Processing: BiLSTM uses bidirectional LSTM for sequential processing\n"
    report += "  - Attention Mechanism: Both use attention, but with different architectures\n"
    report += "  - Parameter Efficiency: Compare parameter counts and distribution\n"
    
    return report
