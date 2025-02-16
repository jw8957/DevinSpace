import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from RephraseModel.models.attention_only import ContentFilterModel
from RephraseModel.models.lstm_attention import LSTMAttentionModel
from RephraseModel.data_processor import ContentDataset
from RephraseModel.train_config import TrainingConfig
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=['Remove', 'Keep'])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    logger.info('\nClassification Report:')
    logger.info(report)
    logger.info('\nConfusion Matrix:')
    logger.info(conf_matrix)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': (np.array(all_preds) == np.array(all_labels)).mean()
    }

def main():
    # Load config
    config = TrainingConfig()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Initialize models
    models = {
        'attention': ContentFilterModel(config.model_name),
        'lstm_attention': LSTMAttentionModel(config.model_name)
    }
    
    # Load test data
    test_dataset = ContentDataset(config.test_file, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Evaluate each model
    results = {}
    for model_type, model in models.items():
        model_path = Path(config.model_dir) / f'best_model_{model_type}.pt'
        if model_path.exists():
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            metrics = evaluate_model(model, test_loader, device)
            results[model_type] = metrics
    
    # Save results
    output_file = Path(config.model_dir) / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
