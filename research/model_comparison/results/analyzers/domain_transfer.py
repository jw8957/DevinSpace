import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def identify_domain(text):
    """Identify the domain/category of the webpage"""
    domains = {
        'news': ['article', 'news', 'report', 'journalist'],
        'ecommerce': ['product', 'price', 'shop', 'cart'],
        'blog': ['blog', 'post', 'author', 'comment'],
        'social': ['profile', 'friend', 'follow', 'share'],
        'documentation': ['docs', 'api', 'reference', 'tutorial']
    }
    
    text_lower = text.lower()
    for domain, keywords in domains.items():
        if any(keyword in text_lower for keyword in keywords):
            return domain
    return 'other'

def analyze_domain_performance(predictions, labels, texts):
    """Analyze model performance across different domains"""
    domain_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        domain = identify_domain(text)
        domain_metrics[domain]['total'] += 1
        if pred == label:
            domain_metrics[domain]['correct'] += 1
    
    # Calculate accuracy per domain
    accuracies = {
        domain: metrics['correct'] / metrics['total']
        for domain, metrics in domain_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate domain distribution
    domain_dist = {
        domain: metrics['total']
        for domain, metrics in domain_metrics.items()
    }
    
    return accuracies, domain_dist

def plot_domain_analysis(bilstm_acc, attn_acc, domain_dist):
    """Plot domain-specific performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by domain
    domains = list(bilstm_acc.keys())
    x = np.arange(len(domains))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[d] for d in domains], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[d] for d in domains], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Domain')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=45)
    ax1.legend()
    
    # Domain distribution
    total = sum(domain_dist.values())
    percentages = {k: v/total*100 for k, v in domain_dist.items()}
    ax2.pie(percentages.values(), labels=percentages.keys(), 
            autopct='%1.1f%%')
    ax2.set_title('Domain Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/domain_analysis.png')
    plt.close()
