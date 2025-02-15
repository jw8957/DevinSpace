import json
import os
from process_metrics import update_metrics, plot_training_progress
from analyze_language_performance import analyze_language_performance, plot_language_comparison
from analyze_efficiency import calculate_efficiency_metrics, plot_efficiency_comparison

def generate_full_report(results, model_params, predictions, labels, languages):
    """Generate comprehensive analysis report"""
    # Update and plot training metrics
    metrics = update_metrics('bilstm', {
        'accuracy': results['accuracy']['bilstm'],
        'train_loss': results['training_time']['bilstm'],
        'val_loss': results['latency']['bilstm']
    })
    plot_training_progress(metrics)
    
    # Analyze language-specific performance
    bilstm_lang = analyze_language_performance(
        predictions['bilstm'], labels['bilstm'], languages['bilstm'])
    attn_lang = analyze_language_performance(
        predictions['attention'], labels['attention'], languages['attention'])
    plot_language_comparison(bilstm_lang, attn_lang)
    
    # Analyze efficiency metrics
    efficiency = calculate_efficiency_metrics(results, model_params)
    plot_efficiency_comparison(efficiency)
    
    # Generate markdown report
    report = f"""# Model Architecture Comparison Report

## Overview
Comparison of BiLSTM+Attention vs Attention-only architectures for content filtering.

### Dataset Statistics
- Total samples: {len(labels['bilstm'])}
- English samples: {languages['bilstm'].count('en')}
- Chinese samples: {languages['bilstm'].count('zh')}

## Performance Metrics

### Overall Accuracy
- BiLSTM+Attention: {results['accuracy']['bilstm']:.4f}
- Attention-only: {results['accuracy']['attention']:.4f}

### Language-Specific Performance
#### BiLSTM+Attention
- English: {bilstm_lang['en']:.4f}
- Chinese: {bilstm_lang['zh']:.4f}

#### Attention-only
- English: {attn_lang['en']:.4f}
- Chinese: {attn_lang['zh']:.4f}

### Computational Efficiency
#### Inference Latency (ms)
- BiLSTM+Attention: {results['latency']['bilstm']*1000:.2f}
- Attention-only: {results['latency']['attention']*1000:.2f}

#### Memory Usage (MB)
- BiLSTM+Attention: {results['memory']['bilstm']/1e6:.2f}
- Attention-only: {results['memory']['attention']/1e6:.2f}

#### Training Time (minutes)
- BiLSTM+Attention: {results['training_time']['bilstm']/60:.2f}
- Attention-only: {results['training_time']['attention']/60:.2f}

## Efficiency Analysis
### BiLSTM+Attention
- Accuracy per parameter: {efficiency['bilstm']['accuracy_per_param']:.2e}
- Accuracy per ms: {efficiency['bilstm']['accuracy_per_ms']:.2e}
- Accuracy per MB: {efficiency['bilstm']['accuracy_per_mb']:.2e}

### Attention-only
- Accuracy per parameter: {efficiency['attention']['accuracy_per_param']:.2e}
- Accuracy per ms: {efficiency['attention']['accuracy_per_ms']:.2e}
- Accuracy per MB: {efficiency['attention']['accuracy_per_mb']:.2e}

## Visualizations
See the `figures/` directory for:
1. Training progress plots
2. Language-specific performance comparison
3. Efficiency metrics comparison

## Recommendations
[Will be added based on experimental results]
"""
    
    with open('../comparison_report.md', 'w') as f:
        f.write(report)
