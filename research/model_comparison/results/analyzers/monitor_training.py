import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class TrainingMonitor:
    def __init__(self, output_dir='../figures'):
        self.output_dir = output_dir
        self.metrics = {
            'bilstm': {
                'train_loss': [],
                'val_loss': [],
                'accuracy': []
            },
            'attention': {
                'train_loss': [],
                'val_loss': [],
                'accuracy': []
            }
        }
        self.last_update = None
    
    def update_metrics(self, model_type, epoch, batch, loss, accuracy=None):
        """Update training metrics"""
        self.metrics[model_type]['train_loss'].append(loss)
        if accuracy is not None:
            self.metrics[model_type]['accuracy'].append(accuracy)
        
        self.last_update = datetime.now()
        
        # Save metrics to file
        self._save_metrics()
        
        # Update visualization if enough new data
        if len(self.metrics[model_type]['train_loss']) % 10 == 0:
            self.plot_progress()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(os.path.join(self.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curves
        for model in ['bilstm', 'attention']:
            if self.metrics[model]['train_loss']:
                ax1.plot(self.metrics[model]['train_loss'], 
                        label=f'{model.capitalize()} Train Loss')
                if self.metrics[model]['val_loss']:
                    ax1.plot(self.metrics[model]['val_loss'],
                            label=f'{model.capitalize()} Val Loss',
                            linestyle='--')
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Updates')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        for model in ['bilstm', 'attention']:
            if self.metrics[model]['accuracy']:
                ax2.plot(self.metrics[model]['accuracy'],
                        label=model.capitalize())
        
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Updates')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_progress.png'))
        plt.close()
    
    def get_latest_metrics(self):
        """Get the most recent metrics"""
        return {
            'bilstm': {
                'train_loss': self.metrics['bilstm']['train_loss'][-1] 
                    if self.metrics['bilstm']['train_loss'] else None,
                'accuracy': self.metrics['bilstm']['accuracy'][-1]
                    if self.metrics['bilstm']['accuracy'] else None
            },
            'attention': {
                'train_loss': self.metrics['attention']['train_loss'][-1]
                    if self.metrics['attention']['train_loss'] else None,
                'accuracy': self.metrics['attention']['accuracy'][-1]
                    if self.metrics['attention']['accuracy'] else None
            }
        }
