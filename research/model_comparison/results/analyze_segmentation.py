import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def identify_segmentation_patterns(text):
    """Identify text segmentation patterns"""
    patterns = {
        'paragraph_breaks': len(re.findall(r'\n\s*\n', text)),
        'bullet_points': len(re.findall(r'^\s*[-•*]\s', text, re.M)),
        'numbered_lists': len(re.findall(r'^\s*\d+\.\s', text, re.M)),
        'section_headers': len(re.findall(r'^[A-Z][^.!?]*[:]\s*$', text, re.M)),
        'continuous_text': len(re.findall(r'[.!?]\s+[A-Z]', text))
    }
    
    # Categorize based on dominant pattern
    max_pattern = max(patterns.items(), key=lambda x: x[1])
    return max_pattern[0] if max_pattern[1] > 0 else 'mixed'

def analyze_segmentation_performance(predictions, labels, texts):
    """Analyze model performance across different segmentation patterns"""
    pattern_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        pattern = identify_segmentation_patterns(text)
        
        pattern_metrics[pattern]['total'] += 1
        if pred == label:
            pattern_metrics[pattern]['correct'] += 1
    
    # Calculate accuracy per pattern
    accuracies = {
        pattern: metrics['correct'] / metrics['total']
        for pattern, metrics in pattern_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate pattern distribution
    pattern_dist = {
        pattern: metrics['total']
        for pattern, metrics in pattern_metrics.items()
    }
    
    return accuracies, pattern_dist

def plot_segmentation_analysis(bilstm_acc, attn_acc, pattern_dist):
    """Plot segmentation-specific performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by pattern
    patterns = list(bilstm_acc.keys())
    x = np.arange(len(patterns))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[p] for p in patterns], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[p] for p in patterns], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Segmentation Pattern')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('_', ' ').title() for p in patterns], 
                        rotation=45)
    ax1.legend()
    
    # Pattern distribution
    total = sum(pattern_dist.values())
    percentages = {k: v/total*100 for k, v in pattern_dist.items()}
    ax2.pie(percentages.values(), 
            labels=[k.replace('_', ' ').title() for k in percentages.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('Segmentation Pattern Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/segmentation_analysis.png')
    plt.close()
