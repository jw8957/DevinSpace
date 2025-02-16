import torch
from torch.utils.data import Dataset
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        """Initialize dataset with data file and tokenizer.
        
        Args:
            data_file: Path to JSONL data file
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length
        """
        logger.info(f"Initializing ContentDataset with file: {data_file}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        
        try:
            # Load and process data
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Original text contains potential boilerplate
                    orig_text = item['origin']
                    # Rephrased text has boilerplate removed
                    clean_text = item['rephrase_with_img']
                    
                    # Split texts into sentences
                    orig_sents = self._split_sentences(orig_text)
                    clean_sents = self._split_sentences(clean_text)
                    
                    # Process each original sentence
                    for sent in orig_sents:
                        sent_text = sent.strip()
                        if not sent_text:
                            continue
                        # Label is 1 if sentence appears in cleaned text (keep)
                        # Label is 0 if sentence was removed (filter)
                        label = 1 if self._is_sentence_kept(sent_text, clean_sents) else 0
                        self.data.append(sent_text)
                        self.labels.append(label)
            
            logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        except Exception as e:
            logger.error(f"Error loading data from {data_file}: {str(e)}")
            raise
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = []
        for line in text.split('\n'):
            if not line.strip():
                continue
            # Split on common sentence endings
            for sent in line.split('. '):
                if sent.strip():
                    sentences.append(sent.strip())
        return sentences
    
    def _is_sentence_kept(self, sent: str, clean_sents: List[str], threshold: float = 0.8) -> bool:
        """Check if sentence appears in cleaned text."""
        sent_tokens = set(sent.lower().split())
        if not sent_tokens:
            return False
        
        for clean_sent in clean_sents:
            clean_tokens = set(clean_sent.lower().split())
            if not clean_tokens:
                continue
            
            # Calculate token overlap
            overlap = len(sent_tokens.intersection(clean_tokens))
            similarity = overlap / max(len(sent_tokens), len(clean_tokens))
            
            if similarity >= threshold:
                return True
        
        return False
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        text = self.data[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
