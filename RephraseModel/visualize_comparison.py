import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison_results(results):
    """Create visualization plots comparing architecture performance"""
    metrics = ['accuracy', 'latency', 'memory', 'training_time']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        data = [results[metric]['bilstm'], 
                results[metric]['attention']]
        
        # Create bar plot
        sns.barplot(x=['BiLSTM+Attention', 'Attention-only'], 
                   y=data, ax=ax)
        
        # Customize plot
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylabel(get_metric_label(metric))
        
        # Add value labels on bars
        for i, v in enumerate(data):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.suptitle('Architecture Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def get_metric_label(metric):
    """Return appropriate label for each metric"""
    labels = {
        'accuracy': 'Accuracy Score',
        'latency': 'Inference Time (ms)',
        'memory': 'Memory Usage (MB)',
        'training_time': 'Training Time (min)'
    }
    return labels.get(metric, metric.capitalize())

def plot_training_progress(history):
    """Plot training metrics over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['bilstm_loss'], label='BiLSTM+Attention')
    ax1.plot(history['attention_loss'], label='Attention-only')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['bilstm_acc'], label='BiLSTM+Attention')
    ax2.plot(history['attention_acc'], label='Attention-only')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Training Progress Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('training_progress.png', bbox_inches='tight', dpi=300)
    plt.close()
