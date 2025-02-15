import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def identify_edge_cases(text, pred, label):
    """Identify edge cases and failure modes"""
    cases = {
        'very_short': len(text.split()) < 5,
        'very_long': len(text.split()) > 100,
        'mixed_language': any(ord(c) > 127 for c in text) and any(ord(c) <= 127 for c in text),
        'special_chars': sum(not c.isalnum() and not c.isspace() for c in text) > len(text) * 0.2,
        'numbers_heavy': sum(c.isdigit() for c in text) > len(text) * 0.1
    }
    return {case: present for case, present in cases.items()}

def analyze_edge_case_performance(predictions, labels, texts):
    """Analyze model performance on edge cases"""
    edge_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label, text in zip(predictions, labels, texts):
        cases = identify_edge_cases(text, pred, label)
        
        for case, present in cases.items():
            if present:
                edge_metrics[case]['total'] += 1
                if pred == label:
                    edge_metrics[case]['correct'] += 1
    
    # Calculate accuracy per edge case
    accuracies = {
        case: metrics['correct'] / metrics['total']
        for case, metrics in edge_metrics.items()
        if metrics['total'] > 0
    }
    
    # Calculate case distribution
    case_dist = {
        case: metrics['total']
        for case, metrics in edge_metrics.items()
    }
    
    return accuracies, case_dist

def plot_edge_case_analysis(bilstm_acc, attn_acc, case_dist):
    """Plot edge case performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by edge case
    cases = list(bilstm_acc.keys())
    x = np.arange(len(cases))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[c] for c in cases], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[c] for c in cases], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance on Edge Cases')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in cases], 
                        rotation=45)
    ax1.legend()
    
    # Edge case distribution
    total = sum(case_dist.values())
    percentages = {k: v/total*100 for k, v in case_dist.items()}
    ax2.pie(percentages.values(), 
            labels=[k.replace('_', ' ').title() for k in percentages.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('Edge Case Distribution')
    
    plt.tight_layout()
    plt.savefig('../figures/edge_case_analysis.png')
    plt.close()
