import torch
from typing import Dict, Any

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
        'predictions': [True, False, True] * 10,  # Sample predictions
        'labels': [True, True, False] * 10,  # Sample labels
        'texts': [
            "This is a main content article.",
            "Home About Contact",
            "Share on social media"
        ] * 10,
        'languages': ['en', 'zh', 'en'] * 10,
        'metrics': {
            'accuracy': {'bilstm': 0.85, 'attention': 0.82},
            'latency': {'bilstm': 0.015, 'attention': 0.012},
            'memory': {'bilstm': 256e6, 'attention': 200e6},
            'training_time': {'bilstm': 3600, 'attention': 3000}
        },
        'outputs': {
            'hidden_states': torch.randn(30, 128),  # Sample hidden states
            'attention_weights': torch.randn(30, 30)  # Sample attention weights
        },
        'device': 'cpu',
        'tokenizer': None  # Placeholder for tokenizer
    }
