import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def identify_text_style(text):
    """Identify the style/register of text"""
    styles = {
        'formal': ['therefore', 'furthermore', 'consequently', 'accordingly'],
        'technical': ['specification', 'documentation', 'implementation', 'procedure'],
        'informal': ['hey', 'yeah', 'cool', 'awesome', 'btw'],
        'promotional': ['sale', 'discount', 'offer', 'limited time', 'special'],
        'descriptive': ['features', 'details', 'overview', 'description']
    }
    
    text_lower = text.lower()
    detected_styles = []
    
    for style, markers in styles.items():
        if any(marker in text_lower for marker in markers):
            detected_styles.append(style)
    
    return detected_styles or ['neutral']

def analyze_style_performance(predictions, labels, texts):
    """Analyze model performance across different text styles"""
    style_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        styles = identify_text_style(text)
        
        for style in styles:
            style_metrics[style]['total'] += 1
            if pred == label:
                style_metrics[style]['correct'] += 1
    
    # Calculate accuracy per style
    accuracies = {
        style: metrics['correct'] / metrics['total']
        for style, metrics in style_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate style distribution
    style_dist = {
        style: metrics['total']
        for style, metrics in style_metrics.items()
    }
    
    return accuracies, style_dist

def plot_style_analysis(bilstm_acc, attn_acc, style_dist):
    """Plot style-specific performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by style
    styles = list(bilstm_acc.keys())
    x = np.arange(len(styles))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[s] for s in styles], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[s] for s in styles], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Text Style')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.title() for s in styles], rotation=45)
    ax1.legend()
    
    # Style distribution
    total = sum(style_dist.values())
    percentages = {k: v/total*100 for k, v in style_dist.items()}
    ax2.pie(percentages.values(), 
            labels=[k.title() for k in percentages.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('Text Style Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/style_analysis.png')
    plt.close()
