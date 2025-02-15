import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def calculate_complexity_metrics(text):
    """Calculate various complexity metrics for a text"""
    return {
        'length': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]),
        'special_chars': sum(not c.isalnum() and not c.isspace() for c in text),
        'numbers': sum(c.isdigit() for c in text)
    }

def bin_complexity(metric_value, bins):
    """Assign a complexity bin based on metric value"""
    for i, threshold in enumerate(bins):
        if metric_value <= threshold:
            return i
    return len(bins)

def analyze_performance_by_complexity(predictions, labels, texts):
    """Analyze model performance across different complexity levels"""
    complexity_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Calculate complexity bins
    lengths = [len(text.split()) for text in texts]
    length_bins = np.percentile(lengths, [25, 50, 75])
    
    for pred, label, text in zip(predictions, labels, texts):
        metrics = calculate_complexity_metrics(text)
        complexity_bin = bin_complexity(metrics['length'], length_bins)
        
        complexity_metrics[complexity_bin]['total'] += 1
        if pred == label:
            complexity_metrics[complexity_bin]['correct'] += 1
    
    # Calculate accuracy per bin
    accuracies = {
        bin_idx: metrics['correct'] / metrics['total']
        for bin_idx, metrics in complexity_metrics.items()
    }
    
    return accuracies, length_bins

def plot_complexity_analysis(bilstm_acc, attn_acc, length_bins):
    """Plot performance comparison across complexity levels"""
    plt.figure(figsize=(12, 6))
    
    bins = ['Simple', 'Medium', 'Complex', 'Very Complex']
    x = np.arange(len(bins))
    width = 0.35
    
    plt.bar(x - width/2, [bilstm_acc[i] for i in range(len(bins))], 
            width, label='BiLSTM+Attention')
    plt.bar(x + width/2, [attn_acc[i] for i in range(len(bins))], 
            width, label='Attention-only')
    
    plt.ylabel('Accuracy')
    plt.title('Model Performance by Text Complexity')
    plt.xticks(x, [f'{bins[i]}\n(≤{int(length_bins[i])} words)' if i < len(length_bins) 
                   else f'{bins[i]}\n(>{int(length_bins[-1])} words)' 
                   for i in range(len(bins))])
    plt.legend()
    
    # Add value labels
    for i in range(len(bins)):
        plt.text(i - width/2, bilstm_acc[i], f'{bilstm_acc[i]:.3f}', 
                ha='center', va='bottom')
        plt.text(i + width/2, attn_acc[i], f'{attn_acc[i]:.3f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/complexity_analysis.png')
    plt.close()
