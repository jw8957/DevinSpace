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
                
                # Process text
                text = text.strip()
                if not text:
                    continue
                
                outputs = model(input_ids, attention_mask)
                predicted = outputs.argmax(dim=1).item()
                
                correct += (predicted == label)
                total += 1
                
                # Calculate confidence
                probs = torch.softmax(outputs, dim=1)
                confidence = probs[0][predicted].item()
                
                results.append({
                    'text': text,
                    'predicted': predicted,
                    'actual': label,
                    'confidence': confidence,
                    'method': model.__class__.__name__
                })
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {str(e)}")
                logger.error(f"Text: {text if 'text' in locals() else 'unknown'}")
                continue
    
    if total == 0:
        logger.warning("No valid samples found for evaluation")
        return 0.0, results
        
    accuracy = 100. * correct / total
    logger.info(f'Accuracy: {accuracy:.2f}% ({correct}/{total} samples)')
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
        # Load tokenizer with multilingual support
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        tokenizer.model_max_length = 512  # Set max length for both languages
        
        # Initialize models
        attention_model = ContentFilterModel().to(device)
        lstm_model = LSTMAttentionModel().to(device)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return
    
    # Test file with mixed languages
    test_file = '/tmp/test_data.json'
    
    # Evaluate on mixed language data
    logger.info("\nEvaluating models on mixed language data:")
    dataset = ContentDataset(test_file, tokenizer)
    
    # Initialize results
    results = {
        'mixed_language': {
            'attention': {},
            'lstm': {},
            'rule_based': {}
        }
    }
    
    # Test attention model
    logger.info("\nTesting Attention-only model:")
    attn_acc, attn_results = evaluate_model_performance(attention_model, dataset, device)
    results['mixed_language']['attention'] = {
        'accuracy': attn_acc,
        'results': attn_results
    }
    
    # Test LSTM model
    logger.info("\nTesting LSTM+Attention model:")
    lstm_acc, lstm_results = evaluate_model_performance(lstm_model, dataset, device)
    results['mixed_language']['lstm'] = {
        'accuracy': lstm_acc,
        'results': lstm_results
    }
    
    # Test rule-based method
    logger.info("\nTesting Rule-based method:")
    rule_results = evaluate_rule_based(dataset)
    results['mixed_language']['rule_based'] = {
        'results': rule_results
    }
    
    # Save results
    try:
        output_dir = Path('RephraseModel')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'evaluation_results.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
    
    return results

if __name__ == '__main__':
    main()
