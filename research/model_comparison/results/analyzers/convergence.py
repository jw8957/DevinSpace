import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_convergence(metrics):
    """Analyze training convergence behavior"""
    convergence_metrics = {}
    for model in ['bilstm', 'attention']:
        train_loss = metrics['training_loss'][model]
        val_loss = metrics['validation_loss'][model]
        
        # Calculate convergence speed
        min_val_loss = min(val_loss)
        min_val_epoch = val_loss.index(min_val_loss)
        
        # Calculate loss stability
        loss_stability = np.std(val_loss[max(0, min_val_epoch-2):min_val_epoch+1])
        
        convergence_metrics[model] = {
            'epochs_to_best': min_val_epoch + 1,
            'best_val_loss': min_val_loss,
            'loss_stability': loss_stability,
            'final_train_loss': train_loss[-1],
            'generalization_gap': train_loss[-1] - val_loss[-1]
        }
    return convergence_metrics

def plot_convergence_analysis(metrics):
    """Generate convergence analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curves
    for model in ['bilstm', 'attention']:
        axes[0, 0].plot(metrics[model]['training_loss'], 
                       label=f'{model.capitalize()} (train)')
        axes[0, 0].plot(metrics[model]['validation_loss'], 
                       label=f'{model.capitalize()} (val)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Generalization gap
    for model in ['bilstm', 'attention']:
        gap = np.array(metrics[model]['training_loss']) - np.array(metrics[model]['validation_loss'])
        axes[0, 1].plot(gap, label=model.capitalize())
    axes[0, 1].set_title('Generalization Gap')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Train Loss - Val Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Loss stability
    for model in ['bilstm', 'attention']:
        stability = [np.std(metrics[model]['validation_loss'][:i+1]) 
                    for i in range(len(metrics[model]['validation_loss']))]
        axes[1, 0].plot(stability, label=model.capitalize())
    axes[1, 0].set_title('Loss Stability')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss Std Dev')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/convergence_analysis.png')
    plt.close()
