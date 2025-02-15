import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_efficiency_metrics(results, model_params):
    """Calculate efficiency metrics for each model"""
    metrics = {}
    for model in ['bilstm', 'attention']:
        accuracy = results['accuracy'][model]
        params = model_params[model]
        inference_time = results['latency'][model]
        memory = results['memory'][model]
        
        metrics[model] = {
            'accuracy_per_param': accuracy / params,
            'accuracy_per_ms': accuracy / (inference_time * 1000),
            'accuracy_per_mb': accuracy / (memory / 1e6)
        }
    return metrics

def plot_efficiency_comparison(metrics):
    """Plot efficiency metrics comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['accuracy_per_param', 'accuracy_per_ms', 'accuracy_per_mb']
    titles = ['Accuracy per Parameter', 'Accuracy per ms', 'Accuracy per MB']
    
    for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        data = [
            metrics['bilstm'][metric],
            metrics['attention'][metric]
        ]
        sns.barplot(
            x=['BiLSTM+Attention', 'Attention-only'],
            y=data,
            ax=axes[idx]
        )
        axes[idx].set_title(title)
        axes[idx].set_ylabel('Efficiency')
        
        # Add value labels
        for i, v in enumerate(data):
            axes[idx].text(i, v, f'{v:.2e}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/efficiency_comparison.png')
    plt.close()
