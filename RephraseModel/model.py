import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F

class ContentFilterModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", dropout=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Bidirectional LSTM for context
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,  # Bidirectional will double this
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Binary classification
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer embeddings
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply LSTM
        lstm_out, _ = self.lstm(sequence_output)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, 
                                   key_padding_mask=~attention_mask.bool())
        
        # Apply classification to each token
        logits = self.classifier(attn_out)
        
        return logits
