import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_language_performance(predictions, labels, languages):
    """Analyze model performance by language"""
    results = {
        'en': {'correct': 0, 'total': 0},
        'zh': {'correct': 0, 'total': 0}
    }
    
    for pred, label, lang in zip(predictions, labels, languages):
        results[lang]['total'] += 1
        if pred == label:
            results[lang]['correct'] += 1
    
    accuracies = {
        lang: results[lang]['correct'] / results[lang]['total']
        for lang in results
    }
    return accuracies

def plot_language_comparison(bilstm_results, attention_results):
    """Plot language-specific performance comparison"""
    languages = ['English', 'Chinese']
    bilstm_acc = [bilstm_results['en'], bilstm_results['zh']]
    attn_acc = [attention_results['en'], attention_results['zh']]
    
    x = np.arange(len(languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, bilstm_acc, width, label='BiLSTM+Attention')
    ax.bar(x + width/2, attn_acc, width, label='Attention-only')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance by Language')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(bilstm_acc):
        ax.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(attn_acc):
        ax.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/language_performance.png')
    plt.close()
