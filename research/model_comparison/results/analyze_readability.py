import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def calculate_readability_metrics(text):
    """Calculate readability metrics for text"""
    # Split into sentences and words
    sentences = re.split(r'[.!?]+', text)
    words = text.split()
    
    # Basic metrics
    avg_sentence_length = len(words) / max(1, len(sentences))
    avg_word_length = sum(len(word) for word in words) / max(1, len(words))
    
    # Categorize readability
    if avg_sentence_length > 20 or avg_word_length > 6:
        return 'complex'
    elif avg_sentence_length > 12 or avg_word_length > 5:
        return 'moderate'
    else:
        return 'simple'

def analyze_readability_performance(predictions, labels, texts):
    """Analyze model performance across different readability levels"""
    readability_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        readability = calculate_readability_metrics(text)
        
        readability_metrics[readability]['total'] += 1
        if pred == label:
            readability_metrics[readability]['correct'] += 1
    
    # Calculate accuracy per readability level
    accuracies = {
        level: metrics['correct'] / metrics['total']
        for level, metrics in readability_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate readability distribution
    level_dist = {
        level: metrics['total']
        for level, metrics in readability_metrics.items()
    }
    
    return accuracies, level_dist

def plot_readability_analysis(bilstm_acc, attn_acc, level_dist):
    """Plot readability-specific performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by readability level
    levels = list(bilstm_acc.keys())
    x = np.arange(len(levels))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[l] for l in levels], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[l] for l in levels], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Readability Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.title() for l in levels])
    ax1.legend()
    
    # Readability distribution
    total = sum(level_dist.values())
    percentages = {k: v/total*100 for k, v in level_dist.items()}
    ax2.pie(percentages.values(), 
            labels=[k.title() for k in percentages.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('Readability Level Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/readability_analysis.png')
    plt.close()
