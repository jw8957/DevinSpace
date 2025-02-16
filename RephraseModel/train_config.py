from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    # Model configuration
    model_type: str = 'attention'  # 'attention' or 'lstm_attention'
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    patience: int = 3
    
    # Data configuration
    max_length: int = 512
    train_file: str = 'RephraseModel/data/train.json'
    val_file: str = 'RephraseModel/data/test.json'  # Using test file as val for now
    test_file: str = 'RephraseModel/data/test.json'
    
    # Model paths
    model_dir: str = 'models'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'max_length': self.max_length
        }
