import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def update_metrics(model_type, epoch_metrics):
    """Update metrics file with new epoch results"""
    metrics_file = 'metrics.json'
    if not os.path.exists(metrics_file):
        with open('metrics_template.json', 'r') as f:
            metrics = json.load(f)
    else:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    # Update metrics
    metrics[model_type]['accuracy'].append(epoch_metrics['accuracy'])
    metrics[model_type]['training_loss'].append(epoch_metrics['train_loss'])
    metrics[model_type]['validation_loss'].append(epoch_metrics['val_loss'])
    
    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def plot_training_progress(metrics):
    """Generate training progress plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for model in ['bilstm', 'attention']:
        if len(metrics[model]['training_loss']) > 0:
            ax1.plot(metrics[model]['training_loss'], 
                    label=f'{model.capitalize()} (train)')
            ax1.plot(metrics[model]['validation_loss'], 
                    label=f'{model.capitalize()} (val)')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    for model in ['bilstm', 'attention']:
        if len(metrics[model]['accuracy']) > 0:
            ax2.plot(metrics[model]['accuracy'], 
                    label=model.capitalize())
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/training_progress.png')
    plt.close()
