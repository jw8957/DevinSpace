import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def categorize_content(text):
    """Categorize content type based on text patterns"""
    categories = {
        'navigation': ['menu', 'navigation', 'breadcrumb', 'sitemap'],
        'social': ['share', 'follow', 'like', 'social'],
        'metadata': ['author', 'date', 'posted', 'comments'],
        'advertisement': ['sponsored', 'advertisement', 'promoted'],
        'related': ['related', 'recommended', 'similar', 'you might like']
    }
    
    text_lower = text.lower()
    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return 'other'

def analyze_content_type_performance(predictions, labels, texts):
    """Analyze model performance across different content types"""
    performance = {}
    type_counts = {}
    
    for pred, label, text in zip(predictions, labels, texts):
        content_type = categorize_content(text)
        
        if content_type not in performance:
            performance[content_type] = {'correct': 0, 'total': 0}
            type_counts[content_type] = 0
        
        performance[content_type]['total'] += 1
        type_counts[content_type] += 1
        if pred == label:
            performance[content_type]['correct'] += 1
    
    # Calculate accuracy per type
    accuracies = {
        ctype: perf['correct'] / perf['total']
        for ctype, perf in performance.items()
    }
    
    return accuracies, type_counts

def plot_type_performance(bilstm_acc, attn_acc, type_counts):
    """Plot performance comparison across content types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by type
    types = list(bilstm_acc.keys())
    x = np.arange(len(types))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[t] for t in types], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[t] for t in types], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Content Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=45)
    ax1.legend()
    
    # Content type distribution
    ax2.pie(type_counts.values(), labels=type_counts.keys(), 
            autopct='%1.1f%%')
    ax2.set_title('Content Type Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/content_type_analysis.png')
    plt.close()
