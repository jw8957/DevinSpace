import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_cross_lingual_transfer(model_results, languages):
    """Analyze how well models transfer between languages"""
    transfer_metrics = {
        'en_to_zh': {
            'accuracy': [],
            'false_positives': [],
            'false_negatives': []
        },
        'zh_to_en': {
            'accuracy': [],
            'false_positives': [],
            'false_negatives': []
        }
    }
    
    # Split data by language
    en_indices = [i for i, lang in enumerate(languages) if lang == 'en']
    zh_indices = [i for i, lang in enumerate(languages) if lang == 'zh']
    
    for model_name, results in model_results.items():
        preds = np.array(results['predictions'])
        labels = np.array(results['labels'])
        
        # Calculate metrics for English training effect on Chinese predictions
        en_trained_zh_preds = preds[zh_indices]
        zh_labels = labels[zh_indices]
        transfer_metrics['en_to_zh']['accuracy'].append(
            np.mean(en_trained_zh_preds == zh_labels))
        
        # Calculate metrics for Chinese training effect on English predictions
        zh_trained_en_preds = preds[en_indices]
        en_labels = labels[en_indices]
        transfer_metrics['zh_to_en']['accuracy'].append(
            np.mean(zh_trained_en_preds == en_labels))
    
    return transfer_metrics

def plot_cross_lingual_analysis(transfer_metrics):
    """Plot cross-lingual transfer analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot EN->ZH transfer
    models = ['BiLSTM+Attention', 'Attention-only']
    en_zh_acc = transfer_metrics['en_to_zh']['accuracy']
    ax1.bar(models, en_zh_acc)
    ax1.set_title('English to Chinese Transfer')
    ax1.set_ylabel('Accuracy')
    for i, v in enumerate(en_zh_acc):
        ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot ZH->EN transfer
    zh_en_acc = transfer_metrics['zh_to_en']['accuracy']
    ax2.bar(models, zh_en_acc)
    ax2.set_title('Chinese to English Transfer')
    ax2.set_ylabel('Accuracy')
    for i, v in enumerate(zh_en_acc):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/cross_lingual_transfer.png')
    plt.close()

def plot_language_confusion_matrices(model_results, languages):
    """Plot confusion matrices for each language pair"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (model_name, results) in enumerate(model_results.items()):
        preds = np.array(results['predictions'])
        labels = np.array(results['labels'])
        
        # English confusion matrix
        en_mask = np.array(languages) == 'en'
        cm_en = confusion_matrix(labels[en_mask], preds[en_mask])
        sns.heatmap(cm_en, annot=True, fmt='d', ax=axes[idx, 0],
                    xticklabels=['Keep', 'Filter'],
                    yticklabels=['Keep', 'Filter'])
        axes[idx, 0].set_title(f'{model_name} - English')
        
        # Chinese confusion matrix
        zh_mask = np.array(languages) == 'zh'
        cm_zh = confusion_matrix(labels[zh_mask], preds[zh_mask])
        sns.heatmap(cm_zh, annot=True, fmt='d', ax=axes[idx, 1],
                    xticklabels=['Keep', 'Filter'],
                    yticklabels=['Keep', 'Filter'])
        axes[idx, 1].set_title(f'{model_name} - Chinese')
    
    plt.tight_layout()
    plt.savefig('../figures/language_confusion_matrices.png')
    plt.close()
