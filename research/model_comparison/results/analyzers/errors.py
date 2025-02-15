import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def analyze_error_patterns(predictions, labels, texts):
    """Analyze error patterns in model predictions"""
    errors = []
    for pred, label, text in zip(predictions, labels, texts):
        if pred != label:
            errors.append({
                'text': text,
                'predicted': pred,
                'actual': label
            })
    return errors

def plot_confusion_matrices(bilstm_results, attention_results):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BiLSTM confusion matrix
    cm_bilstm = confusion_matrix(
        bilstm_results['true'], 
        bilstm_results['pred']
    )
    sns.heatmap(cm_bilstm, annot=True, fmt='d', ax=ax1,
                xticklabels=['Keep', 'Filter'],
                yticklabels=['Keep', 'Filter'])
    ax1.set_title('BiLSTM+Attention')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Attention-only confusion matrix
    cm_attn = confusion_matrix(
        attention_results['true'],
        attention_results['pred']
    )
    sns.heatmap(cm_attn, annot=True, fmt='d', ax=ax2,
                xticklabels=['Keep', 'Filter'],
                yticklabels=['Keep', 'Filter'])
    ax2.set_title('Attention-only')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../figures/confusion_matrices.png')
    plt.close()

def analyze_sequence_length_impact(predictions, labels, lengths):
    """Analyze impact of sequence length on model performance"""
    correct = np.array(predictions) == np.array(labels)
    length_bins = np.percentile(lengths, [0, 25, 50, 75, 100])
    accuracy_by_length = []
    
    for i in range(len(length_bins)-1):
        mask = (lengths >= length_bins[i]) & (lengths < length_bins[i+1])
        accuracy = correct[mask].mean()
        accuracy_by_length.append(accuracy)
    
    return accuracy_by_length, length_bins[:-1]

def plot_length_analysis(bilstm_acc, attn_acc, length_bins):
    """Plot accuracy vs sequence length"""
    plt.figure(figsize=(10, 6))
    plt.plot(length_bins, bilstm_acc, 'o-', label='BiLSTM+Attention')
    plt.plot(length_bins, attn_acc, 'o-', label='Attention-only')
    plt.xlabel('Sequence Length Percentile')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Sequence Length')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figures/length_analysis.png')
    plt.close()
