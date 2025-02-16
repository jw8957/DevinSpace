import torch
import torch.nn as nn
from transformers import AutoModel

class ContentFilterModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Attention layer for sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=384,  # Base model hidden size
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)  # Binary classification
        )
    
    def forward(self, input_ids, attention_mask):
        # Get transformer embeddings
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        # Apply attention mechanism
        attn_output, _ = self.attention(
            hidden_states, 
            hidden_states, 
            hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Get sequence representation
        sequence_output = attn_output.mean(dim=1)
        
        # Classify
        logits = self.classifier(sequence_output)
        return logits
