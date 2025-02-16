import torch
from transformers import AutoTokenizer
import logging
from pathlib import Path
from RephraseModel.models import ContentFilterModel, LSTMAttentionModel
from RephraseModel.data_processor import ContentDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading and processing."""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test with first 10 lines only
    def read_first_n_lines(file_path, n=10):
        with open(file_path, 'r') as f:
            lines = [next(f) for _ in range(n)]
        with open('/tmp/test_data.json', 'w') as f:
            f.writelines(lines)
        return '/tmp/test_data.json'
    
    # Test English data
    en_file = 'RephraseModel/data/train.json'
    test_file = read_first_n_lines(en_file)
    logger.info(f"Testing data loading with first 10 lines from {en_file}")
    en_dataset = ContentDataset(test_file, tokenizer, max_length=512)
    logger.info(f"Loaded {len(en_dataset)} samples")
    
    # Test batch creation
    batch_size = 4
    sample_batch = [en_dataset[i] for i in range(min(batch_size, len(en_dataset)))]
    logger.info(f"Successfully created batch of {len(sample_batch)} samples")
    
    # Verify batch contents
    for i, item in enumerate(sample_batch):
        logger.info(f"Sample {i}:")
        logger.info(f"Input shape: {item['input_ids'].shape}")
        logger.info(f"Label: {item['labels']}")

def test_model_forward():
    """Test model forward pass."""
    # Initialize models
    attention_model = ContentFilterModel()
    lstm_model = LSTMAttentionModel()
    
    # Create dummy input
    batch_size = 4
    seq_length = 128
    dummy_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length)
    }
    
    # Test forward pass
    logger.info("Testing Attention-only model")
    with torch.no_grad():
        attention_output = attention_model(**dummy_input)
    logger.info(f"Attention model output shape: {attention_output.shape}")
    
    logger.info("Testing LSTM+Attention model")
    with torch.no_grad():
        lstm_output = lstm_model(**dummy_input)
    logger.info(f"LSTM model output shape: {lstm_output.shape}")

if __name__ == '__main__':
    try:
        logger.info("Starting data loading test")
        test_data_loading()
        logger.info("Data loading test completed successfully")
        
        logger.info("\nStarting model forward pass test")
        test_model_forward()
        logger.info("Model forward pass test completed successfully")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
