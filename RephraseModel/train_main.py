import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
from pathlib import Path
from typing import Optional

from models import ContentFilterModel, LSTMAttentionModel
from data_processor import ContentDataset
from trainer import ModelTrainer
from train_config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model(config: TrainingConfig):
    if config.model_type == 'attention':
        return ContentFilterModel(config.model_name)
    elif config.model_type == 'lstm_attention':
        return LSTMAttentionModel(config.model_name)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def train(config: Optional[TrainingConfig] = None):
    if config is None:
        config = TrainingConfig()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = get_model(config)
    
    # Prepare data
    train_dataset = ContentDataset(config.train_file, tokenizer, config.max_length)
    val_dataset = ContentDataset(config.val_file, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Setup trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_dir=Path(config.model_dir),
        config=config.to_dict()
    )
    
    # Train model
    trainer.train()
    logger.info('Training completed')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['attention', 'lstm_attention'],
                       default='attention', help='Model architecture to use')
    args = parser.parse_args()
    
    config = TrainingConfig(model_type=args.model_type)
    train(config)
