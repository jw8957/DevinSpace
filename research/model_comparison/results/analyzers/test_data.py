import torch
from typing import Dict, Any
from transformers import AutoTokenizer

def create_test_model(model_type: str) -> torch.nn.Module:
    """Create a test model with realistic architecture"""
    if model_type == 'bilstm':
        model = torch.nn.Sequential(
            torch.nn.LSTM(input_size=768, hidden_size=256, bidirectional=True, batch_first=True),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
    else:  # attention
        model = torch.nn.Sequential(
            torch.nn.MultiheadAttention(embed_dim=768, num_heads=8),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
    return model

def get_test_data() -> Dict[str, Any]:
    """Generate test data for analysis pipeline development"""
    return {
        'bilstm_model': create_test_model('bilstm'),
        'attention_model': create_test_model('attention'),
        'predictions': [1, 0, 1] * 10,  # Sample predictions (1=keep, 0=filter)
        'labels': [1, 1, 0] * 10,  # Sample labels (1=keep, 0=filter)
        'texts': [
            "This is a main content article.",
            "Home About Contact",
            "Share on social media"
        ] * 10,
        'original_texts': [
            "This is a main content article with some extra navigation.",
            "Home About Contact Products Services",
            "Share on social media Follow us Subscribe"
        ] * 10,
        'languages': ['en', 'zh', 'en'] * 10,
        'metrics': {
            'bilstm': {
                'accuracy': 0.85,
                'latency': 0.015,
                'memory': 256e6,
                'training_time': 3600,
                'training_loss': [2.5, 1.8, 1.2, 0.9, 0.7],
                'validation_loss': [2.6, 1.9, 1.3, 1.0, 0.8]
            },
            'attention': {
                'accuracy': 0.82,
                'latency': 0.012,
                'memory': 200e6,
                'training_time': 3000,
                'training_loss': [2.3, 1.7, 1.1, 0.8, 0.6],
                'validation_loss': [2.4, 1.8, 1.2, 0.9, 0.7]
            }
        },
        'outputs': {
            'hidden_states': torch.randn(30, 128),  # Sample hidden states
            'attention_weights': torch.randn(30, 30)  # Sample attention weights
        },
        'attention_weights': torch.randn(30, 30),  # Sample attention weights for analysis
        'device': 'cpu',
        'tokenizer': AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    }
