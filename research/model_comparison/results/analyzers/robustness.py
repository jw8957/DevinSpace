import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import defaultdict

def predict(model, tokenizer, text, device):
    """Make a single prediction"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=-1)
        # Return prediction for first (and only) sequence
        return predictions[0].cpu().numpy()

def generate_variations(text):
    """Generate text variations to test model robustness"""
    variations = []
    
    # Truncation
    words = text.split()
    if len(words) > 5:
        variations.append(' '.join(words[:len(words)//2]))
    
    # Extra whitespace
    variations.append(' '.join([w + '  ' for w in words]))
    
    # Character noise
    noisy = []
    for word in words:
        if len(word) > 3:
            idx = np.random.randint(1, len(word)-1)
            noisy.append(word[:idx] + word[idx+1:])
        else:
            noisy.append(word)
    variations.append(' '.join(noisy))
    
    return variations

def analyze_robustness(model, tokenizer, texts, device):
    """Analyze model robustness to input variations"""
    results = defaultdict(list)
    
    for text in texts:
        base_pred = predict(model, tokenizer, text, device)
        variations = generate_variations(text)
        
        # Test variations
        variation_preds = [predict(model, tokenizer, var, device) 
                         for var in variations]
        
        # Calculate consistency
        consistency = np.mean([pred == base_pred for pred in variation_preds])
        results['consistency'].append(consistency)
        
        # Track variation types
        results['truncation'].append(variation_preds[0] == base_pred)
        results['whitespace'].append(variation_preds[1] == base_pred)
        results['char_noise'].append(variation_preds[2] == base_pred)
    
    return results

def plot_robustness_analysis(bilstm_results, attn_results):
    """Plot robustness analysis results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall consistency
    models = ['BiLSTM+Attention', 'Attention-only']
    consistency = [
        np.mean(bilstm_results['consistency']),
        np.mean(attn_results['consistency'])
    ]
    
    ax1.bar(models, consistency)
    ax1.set_title('Overall Prediction Consistency')
    ax1.set_ylabel('Consistency Score')
    for i, v in enumerate(consistency):
        ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Variation type analysis
    variation_types = ['Truncation', 'Whitespace', 'Character Noise']
    bilstm_scores = [
        np.mean(bilstm_results['truncation']),
        np.mean(bilstm_results['whitespace']),
        np.mean(bilstm_results['char_noise'])
    ]
    attn_scores = [
        np.mean(attn_results['truncation']),
        np.mean(attn_results['whitespace']),
        np.mean(attn_results['char_noise'])
    ]
    
    x = np.arange(len(variation_types))
    width = 0.35
    ax2.bar(x - width/2, bilstm_scores, width, label='BiLSTM+Attention')
    ax2.bar(x + width/2, attn_scores, width, label='Attention-only')
    ax2.set_title('Robustness by Variation Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(variation_types)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/robustness_analysis.png')
    plt.close()
