import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        model_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'patience': 3,
            'save_best': True
        }
        if config:
            self.config.update(config)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(self.train_loader)
            self.metrics['train_loss'].append(avg_train_loss)
            
            # Validation phase
            val_loss, accuracy = self._validate()
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(accuracy)
            
            logger.info(f'Epoch {epoch+1}/{self.config["num_epochs"]}:')
            logger.info(f'Training Loss: {avg_train_loss:.4f}')
            logger.info(f'Validation Loss: {val_loss:.4f}')
            logger.info(f'Validation Accuracy: {accuracy:.2f}%')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config['save_best']:
                    self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.info('Early stopping triggered')
                    break
            
            # Save metrics
            self._save_metrics()
    
    def _validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def _save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        torch.save(checkpoint, self.model_dir / filename)
    
    def _save_metrics(self):
        metrics_file = self.model_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(self.model_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.metrics = checkpoint['metrics']
