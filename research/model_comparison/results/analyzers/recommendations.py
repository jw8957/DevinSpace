import numpy as np
from typing import Dict, Any

def analyze_performance_tradeoffs(results: Dict[str, Any]) -> Dict[str, str]:
    """Analyze performance tradeoffs between architectures"""
    tradeoffs = {}
    metrics = results.get('metrics', {})
    
    # Accuracy vs Speed
    acc_diff = metrics['accuracy']['bilstm'] - metrics['accuracy']['attention']
    speed_ratio = metrics['latency']['bilstm'] / metrics['latency']['attention']
    
    if acc_diff > 0.02 and speed_ratio > 1.2:
        tradeoffs['accuracy_vs_speed'] = (
            "BiLSTM offers higher accuracy but with increased latency. "
            "Consider using BiLSTM for non-realtime applications where "
            "accuracy is critical."
        )
    elif acc_diff < 0.02 and speed_ratio > 1.2:
        tradeoffs['accuracy_vs_speed'] = (
            "Attention-only model achieves similar accuracy with lower "
            "latency. Prefer this for real-time applications."
        )
    
    # Memory vs Accuracy
    mem_ratio = results['memory']['bilstm'] / results['memory']['attention']
    if mem_ratio > 1.5 and acc_diff < 0.02:
        tradeoffs['memory_vs_accuracy'] = (
            "Attention-only model achieves similar accuracy with lower "
            "memory footprint. Prefer this for resource-constrained "
            "environments."
        )
    
    return tradeoffs

def analyze_language_specific_performance(results: Dict[str, Any]) -> Dict[str, str]:
    """Analyze language-specific performance patterns"""
    recommendations = {}
    
    for lang in ['en', 'zh']:
        bilstm_acc = results['language_metrics']['bilstm'][lang]
        attn_acc = results['language_metrics']['attention'][lang]
        
        if abs(bilstm_acc - attn_acc) > 0.05:
            better_model = "BiLSTM" if bilstm_acc > attn_acc else "Attention-only"
            recommendations[f'{lang}_specific'] = (
                f"{better_model} model performs significantly better for "
                f"{'English' if lang == 'en' else 'Chinese'} content. "
                f"Consider using this model for {lang} specific deployments."
            )
    
    return recommendations

def generate_recommendations(results: Dict[str, Any]) -> str:
    """Generate comprehensive recommendations based on all analyses"""
    tradeoffs = analyze_performance_tradeoffs(results)
    lang_recommendations = analyze_language_specific_performance(results)
    
    report = "# Model Architecture Recommendations\n\n"
    
    # Overall recommendation
    report += "## Primary Recommendation\n"
    metrics = results.get('metrics', {})
    if metrics['accuracy']['bilstm'] > metrics['accuracy']['attention'] * 1.05:
        report += ("Use BiLSTM+Attention model for highest accuracy, "
                  "especially for complex content filtering tasks.\n\n")
    else:
        report += ("Use Attention-only model for better efficiency with "
                  "comparable accuracy.\n\n")
    
    # Performance tradeoffs
    report += "## Performance Tradeoffs\n"
    for tradeoff, description in tradeoffs.items():
        report += f"- {description}\n"
    report += "\n"
    
    # Language-specific recommendations
    report += "## Language-specific Considerations\n"
    for lang, recommendation in lang_recommendations.items():
        report += f"- {recommendation}\n"
    report += "\n"
    
    # Deployment considerations
    report += "## Deployment Considerations\n"
    metrics = results.get('metrics', {})
    report += ("- Resource Requirements:\n"
               f"  - BiLSTM: {metrics['memory']['bilstm']/1e6:.1f}MB memory, "
               f"{metrics['latency']['bilstm']*1000:.1f}ms latency\n"
               f"  - Attention: {metrics['memory']['attention']/1e6:.1f}MB memory, "
               f"{metrics['latency']['attention']*1000:.1f}ms latency\n\n")
    
    return report
