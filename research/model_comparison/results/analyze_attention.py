import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_attention_weights(model, input_ids, attention_mask):
    """Extract attention weights from model forward pass"""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, output_attentions=True)
        # Get attention weights from last layer
        attention = outputs.attentions[-1].mean(dim=1)  # Average over attention heads
    return attention.cpu().numpy()

def plot_attention_heatmap(attention_weights, tokens, save_path):
    """Plot attention weight heatmap"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention_weights, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis')
    plt.title('Attention Weight Distribution')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_attention_patterns(model, tokenizer, examples):
    """Analyze attention patterns for specific examples"""
    patterns = {
        'long_range': [],
        'local': [],
        'uniform': []
    }
    
    for text in examples:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        attention = extract_attention_weights(
            model, 
            inputs['input_ids'], 
            inputs['attention_mask']
        )[0]
        
        # Analyze attention distribution
        avg_dist = np.mean(np.abs(np.arange(attention.shape[0])[:, None] - 
                                 np.arange(attention.shape[1])[None, :]) * attention)
        
        if avg_dist > 0.5:  # Long-range dependencies
            patterns['long_range'].append(text)
        elif np.std(attention) < 0.1:  # Uniform attention
            patterns['uniform'].append(text)
        else:  # Local attention
            patterns['local'].append(text)
    
    return patterns

def plot_pattern_distribution(patterns):
    """Plot distribution of attention patterns"""
    pattern_counts = {k: len(v) for k, v in patterns.items()}
    
    plt.figure(figsize=(10, 6))
    plt.bar(pattern_counts.keys(), pattern_counts.values())
    plt.title('Distribution of Attention Patterns')
    plt.xlabel('Pattern Type')
    plt.ylabel('Count')
    
    # Add value labels
    for i, v in enumerate(pattern_counts.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../figures/attention_patterns.png')
    plt.close()
