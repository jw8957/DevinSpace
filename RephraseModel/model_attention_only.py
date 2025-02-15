import torch
import torch.nn as nn
from transformers import AutoModel

class ContentFilterModelAttentionOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.attention = nn.MultiheadAttention(384, num_heads=8)
        self.classifier = nn.Linear(384, 2)
        
    def forward(self, input_ids, attention_mask):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask)
        sequence_output = outputs[0]
        
        # Apply self-attention
        # Transpose for attention: [batch_size, seq_len, hidden] -> [seq_len, batch_size, hidden]
        attn_output, _ = self.attention(
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1)
        )
        
        # Transpose back: [seq_len, batch_size, hidden] -> [batch_size, seq_len, hidden]
        attn_output = attn_output.transpose(0, 1)
        
        # Classification layer
        logits = self.classifier(attn_output)
        return logits
