import json
import spacy
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import difflib

import logging
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
        
        # Load and process data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Original text contains potential boilerplate
                orig_text = item['origin']
                # Rephrased text has boilerplate removed
                clean_text = item['rephrase_with_img']
                
                # Create binary labels (1 for keep, 0 for remove)
                # by comparing original and cleaned text
                self._process_text_pair(orig_text, clean_text)
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        
    def load_and_process_data(self, data_file: str) -> List[Dict]:
        logger.info("Starting data processing...")
        processed_samples = []
        
        # Load appropriate spaCy model based on file name
        if "zh-hans" in data_file:
            logger.info("Loading Chinese spaCy model...")
            nlp = spacy.load("zh_core_web_sm")
            logger.info("Chinese model loaded successfully")
        else:
            logger.info("Loading English spaCy model...")
            nlp = spacy.load("en_core_web_sm")
            logger.info("English model loaded successfully")
            
        # Configure pipeline for sentence segmentation only
        for pipe in nlp.pipe_names:
            if pipe not in ['sentencizer', 'parser']:
                nlp.disable_pipe(pipe)
                
        # Ensure we have a sentence segmenter
        if 'sentencizer' not in nlp.pipe_names and 'parser' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer')
        
        # Configure spaCy for better performance
        nlp.max_length = 2000000  # Increase max text length
        
        # Process in batches for better performance
        batch_size = 50
        samples = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        logger.info(f"Total lines in file: {total_lines}")
        
        for i in range(0, total_lines, batch_size):
            batch = lines[i:i+batch_size]
            if i % 100 == 0:
                logger.info(f"Processing batch starting at line {i}/{total_lines}")
            
            for line in batch:
                try:
                    data = json.loads(line)
                    orig_text = data['origin']
                    clean_text = data['rephrase_with_img']
                
                    # Skip empty examples
                    if not orig_text.strip() or not clean_text.strip():
                        continue
                        
                    # Split texts into sentences
                    try:
                        orig_doc = nlp(orig_text)
                        clean_doc = nlp(clean_text)
                        
                        # Verify sentence segmentation worked
                        orig_sents = [sent.text.strip() for sent in orig_doc.sents]
                        clean_sents = [sent.text.strip() for sent in clean_doc.sents]
                        
                        if not orig_sents:  # Fallback to simple splitting if no sentences found
                            orig_sents = [s.strip() for s in orig_text.split('\n') if s.strip()]
                        if not clean_sents:
                            clean_sents = [s.strip() for s in clean_text.split('\n') if s.strip()]
                            
                    except Exception as e:
                        logger.warning(f"Sentence segmentation failed, falling back to simple splitting: {str(e)}")
                        orig_sents = [s.strip() for s in orig_text.split('\n') if s.strip()]
                        clean_sents = [s.strip() for s in clean_text.split('\n') if s.strip()]
                    
                    if not orig_sents:  # Skip if no sentences found
                        continue
                    
                    # Create labels using sequence matching
                    labels = self.align_and_label_sentences(orig_sents, clean_sents)
                    
                    # Process each sentence
                    for sent, label in zip(orig_sents, labels):
                        if sent.strip():  # Skip empty sentences
                            processed_samples.append({
                                'text': sent,
                                'label': label
                            })
                            
                    if len(processed_samples) % 100 == 0:
                        logger.info(f"Processed {len(processed_samples)} sentences so far...")
                except Exception as e:
                    logger.error(f"Error processing line: {str(e)}")
                    continue
        
        return processed_samples
    
    def align_and_label_sentences(self, orig_sents: List[str], clean_sents: List[str]) -> List[int]:
        """Use sequence matching to determine which sentences were kept (1) or removed (0)"""
        matcher = difflib.SequenceMatcher(None, orig_sents, clean_sents)
        labels = [0] * len(orig_sents)  # Initialize all as removed
        
        # Mark matching blocks as kept
        for block in matcher.get_matching_blocks():
            for i in range(block.size):
                labels[block.a + i] = 1
                
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create a sequence of labels (one per token)
        token_labels = torch.zeros(self.max_length, dtype=torch.long)
        token_labels[:encoding['attention_mask'].sum()] = item['label']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': token_labels
        }
