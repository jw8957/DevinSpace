import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

def extract_html_context(text):
    """Extract HTML context patterns from text"""
    contexts = {
        'navigation': r'<nav>|<menu>|<ul.*?>|<ol.*?>',
        'content': r'<article>|<section>|<main>|<div.*?class="content"',
        'sidebar': r'<aside>|<div.*?class="sidebar"',
        'header': r'<header>|<div.*?class="header"',
        'footer': r'<footer>|<div.*?class="footer"'
    }
    
    detected = {}
    for context, pattern in contexts.items():
        detected[context] = bool(re.search(pattern, text, re.IGNORECASE))
    return detected

def analyze_html_context_performance(predictions, labels, texts):
    """Analyze model performance across different HTML contexts"""
    context_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        contexts = extract_html_context(text)
        
        for context, present in contexts.items():
            if present:
                context_metrics[context]['total'] += 1
                if pred == label:
                    context_metrics[context]['correct'] += 1
    
    # Calculate accuracy per context
    accuracies = {
        context: metrics['correct'] / metrics['total']
        for context, metrics in context_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate context distribution
    context_dist = {
        context: metrics['total']
        for context, metrics in context_metrics.items()
    }
    
    return accuracies, context_dist

def plot_html_context_analysis(bilstm_acc, attn_acc, context_dist):
    """Plot HTML context performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by context
    contexts = list(bilstm_acc.keys())
    x = np.arange(len(contexts))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[c] for c in contexts], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[c] for c in contexts], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by HTML Context')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.title() for c in contexts], rotation=45)
    ax1.legend()
    
    # Context distribution
    total = sum(context_dist.values())
    percentages = {k: v/total*100 for k, v in context_dist.items()}
    ax2.pie(percentages.values(), 
            labels=[k.title() for k in percentages.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('HTML Context Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/html_context_analysis.png')
    plt.close()
