import matplotlib.pyplot as plt
import re
import numpy as np

def parse_training_log(log_file):
    """Parse training log to extract metrics"""
    epochs = []
    train_losses = []
    val_losses = []
    precisions = []
    recalls = []
    f1_scores = []
    
    current_epoch = None
    with open(log_file, 'r') as f:
        for line in f:
            # Extract epoch
            epoch_match = re.search(r'Starting epoch (\d+)/\d+', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                
            # Extract validation metrics
            if 'Validation Loss:' in line:
                epochs.append(current_epoch)
                val_loss = float(re.search(r'Validation Loss: ([\d.]+)', line).group(1))
                val_losses.append(val_loss)
            
            if 'Precision:' in line:
                precision = float(re.search(r'Precision: ([\d.]+)', line).group(1))
                precisions.append(precision)
            
            if 'Recall:' in line:
                recall = float(re.search(r'Recall: ([\d.]+)', line).group(1))
                recalls.append(recall)
            
            if 'F1 Score:' in line:
                f1 = float(re.search(r'F1 Score: ([\d.]+)', line).group(1))
                f1_scores.append(f1)
    
    return {
        'epochs': epochs,
        'val_losses': val_losses,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores
    }

def plot_training_progress(metrics):
    """Create plots showing training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot validation loss
    ax1.plot(metrics['epochs'], metrics['val_losses'], 'b-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Validation Loss Over Time')
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(metrics['epochs'], metrics['precisions'], 'g-', label='Precision')
    ax2.plot(metrics['epochs'], metrics['recalls'], 'r-', label='Recall')
    ax2.plot(metrics['epochs'], metrics['f1_scores'], 'b-', label='F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Metrics Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    # Parse and plot training progress
    metrics = parse_training_log('training.log')
    plot_training_progress(metrics)
