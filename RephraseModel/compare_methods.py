import torch
from transformers import AutoTokenizer
import logging
from pathlib import Path
import json
import sys
from typing import List, Dict, Any
from RephraseModel.models import ContentFilterModel, LSTMAttentionModel
from RephraseModel.data_processor import ContentDataset

# Add parent directory to path for importing web_processor_V2
sys.path.append(str(Path(__file__).parent.parent))
from RephraseModel.web_processor_V2 import RephraseContent_V2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model_performance(model, dataset, device='cpu'):
    """Evaluate model performance on dataset."""
    model.eval()
    correct = 0
    total = 0
    results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                batch = dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0).to(device)
                attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
                label = batch['labels'].item()
                text = batch['text']
                
                # Skip empty texts
                if not text or text == "empty":
                    continue
                
                outputs = model(input_ids, attention_mask)
                predicted = outputs.argmax(dim=1).item()
                
                correct += (predicted == label)
                total += 1
                
                results.append({
                    'text': text,
                    'predicted': predicted,
                    'actual': label,
                    'method': model.__class__.__name__
                })
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {str(e)}")
                logger.error(f"Text: {text if 'text' in locals() else 'unknown'}")
                continue
    
    if total == 0:
        logger.warning("No valid samples found for evaluation")
        return 0.0, results
        
    accuracy = 100. * correct / total
    logger.info(f'Accuracy: {accuracy:.2f}%')
    return accuracy, results

def evaluate_rule_based(dataset: ContentDataset) -> List[Dict[str, Any]]:
    """Evaluate rule-based method."""
    results = []
    for i in range(len(dataset)):
        try:
            # Get sample from dataset
            sample = dataset[i]
            text = sample['text']
            label = sample['labels'].item()
            
            # Skip if text is empty or "empty" placeholder
            if not text or text == "empty":
                continue
                
            # Clean text for processing
            text = text.strip()
            if not text:
                continue
                
            try:
                # Create processor instance for each text
                processor = RephraseContent_V2(raw_markdown=text)
                # Apply rule-based processing
                processed = processor.process_content()
                # Consider text kept if it appears in processed output
                kept = text in processed
            except Exception as e:
                logger.warning(f"Rule-based processing failed for text: {text[:100]}...")
                logger.warning(f"Error: {str(e)}")
                kept = False  # Default to removing text on processing error
            
            results.append({
                'text': text,
                'predicted': int(kept),
                'actual': label,
                'method': 'rule_based'
            })
        except Exception as e:
            logger.error(f"Error accessing sample {i}: {str(e)}")
            continue
            
    if not results:
        logger.warning("No valid samples processed by rule-based method")
        
    return results

def main():
    # Initialize models and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        attention_model = ContentFilterModel().to(device)
        lstm_model = LSTMAttentionModel().to(device)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return
    
    # Test files
    en_file = '/tmp/test_data.json'  # Use smaller test set
    zh_file = '/tmp/test_data.json'  # Use same test set for now
    
    # Evaluate on English data
    logger.info("\nEvaluating on English data:")
    en_dataset = ContentDataset(en_file, tokenizer)
    
    logger.info("Testing Attention-only model:")
    en_attn_acc, en_attn_results = evaluate_model_performance(attention_model, en_dataset, device)
    
    logger.info("Testing LSTM+Attention model:")
    en_lstm_acc, en_lstm_results = evaluate_model_performance(lstm_model, en_dataset, device)
    
    logger.info("Testing Rule-based method:")
    en_rule_results = evaluate_rule_based(en_dataset)
    
    # Evaluate on Chinese data
    logger.info("\nEvaluating on Chinese data:")
    zh_dataset = ContentDataset(zh_file, tokenizer)
    
    logger.info("Testing Attention-only model:")
    zh_attn_acc, zh_attn_results = evaluate_model_performance(attention_model, zh_dataset, device)
    
    logger.info("Testing LSTM+Attention model:")
    zh_lstm_acc, zh_lstm_results = evaluate_model_performance(lstm_model, zh_dataset, device)
    
    logger.info("Testing Rule-based method:")
    zh_rule_results = evaluate_rule_based(zh_dataset)
    
    # Save results
    results = {
        'english': {
            'attention': {'accuracy': en_attn_acc, 'results': en_attn_results},
            'lstm': {'accuracy': en_lstm_acc, 'results': en_lstm_results},
            'rule_based': {'results': en_rule_results}
        },
        'chinese': {
            'attention': {'accuracy': zh_attn_acc, 'results': zh_attn_results},
            'lstm': {'accuracy': zh_lstm_acc, 'results': zh_lstm_results},
            'rule_based': {'results': zh_rule_results}
        }
    }
    
    with open('RephraseModel/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("\nResults saved to RephraseModel/evaluation_results.json")

if __name__ == '__main__':
    main()
