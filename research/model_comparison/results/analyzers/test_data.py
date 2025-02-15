import torch
from typing import Dict, Any

def get_test_data() -> Dict[str, Any]:
    """Generate test data for analysis pipeline development"""
    return {
        'bilstm_model': torch.nn.Module(),  # Placeholder model
        'attention_model': torch.nn.Module(),  # Placeholder model
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
