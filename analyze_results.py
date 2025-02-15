import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from data_processor import ContentDataset
from model import ContentFilterModel

def load_and_evaluate_model(model_path, test_data_en, test_data_zh):
    """Load trained model and evaluate on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ContentFilterModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create datasets
    en_dataset = ContentDataset(test_data_en)
    zh_dataset = ContentDataset(test_data_zh)
    
    results = {
        'en': evaluate_dataset(model, en_dataset, device),
        'zh': evaluate_dataset(model, zh_dataset, device)
    }
    return results

def evaluate_dataset(model, dataset, device):
    """Evaluate model on a dataset"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)
            
            # Only collect predictions for non-padded tokens
            valid_preds = preds[0][attention_mask[0] == 1].cpu().numpy()
            valid_labels = labels[attention_mask[0] == 1].numpy()
            
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
    
    return {
        'predictions': all_preds,
        'labels': all_labels
    }

def plot_confusion_matrices(results):
    """Plot confusion matrices for both languages"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # English results
    cm_en = confusion_matrix(results['en']['labels'], results['en']['predictions'])
    sns.heatmap(cm_en, annot=True, fmt='d', ax=ax1)
    ax1.set_title('English Content Filtering')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Chinese results
    cm_zh = confusion_matrix(results['zh']['labels'], results['zh']['predictions'])
    sns.heatmap(cm_zh, annot=True, fmt='d', ax=ax2)
    ax2.set_title('Chinese Content Filtering')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def compare_with_rule_based(results, en_file, zh_file):
    """Compare model predictions with rule-based approach"""
    def count_filtered_sentences(file_path):
        total = 0
        filtered = 0
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                orig_sents = data['origin'].split('\n')
                clean_sents = data['rephrase_with_img'].split('\n')
                total += len(orig_sents)
                filtered += len(orig_sents) - len(clean_sents)
        return total, filtered
    
    # Rule-based statistics
    en_total, en_filtered = count_filtered_sentences(en_file)
    zh_total, zh_filtered = count_filtered_sentences(zh_file)
    
    # Model statistics
    model_en_filtered = sum(1 for p in results['en']['predictions'] if p == 0)
    model_zh_filtered = sum(1 for p in results['zh']['predictions'] if p == 0)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # English comparison
    en_data = {
        'Rule-based': en_filtered / en_total * 100,
        'Model': model_en_filtered / len(results['en']['predictions']) * 100
    }
    ax1.bar(en_data.keys(), en_data.values())
    ax1.set_title('English Content Filtering')
    ax1.set_ylabel('Filtered Content (%)')
    
    # Chinese comparison
    zh_data = {
        'Rule-based': zh_filtered / zh_total * 100,
        'Model': model_zh_filtered / len(results['zh']['predictions']) * 100
    }
    ax2.bar(zh_data.keys(), zh_data.values())
    ax2.set_title('Chinese Content Filtering')
    ax2.set_ylabel('Filtered Content (%)')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.close()
    
    return {
        'en': en_data,
        'zh': zh_data
    }

if __name__ == "__main__":
    # Paths
    model_path = 'model_outputs/best_model.pt'
    en_file = '/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl'
    zh_file = '/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl'
    
    # Wait for training to complete and model to be saved
    print("Waiting for training to complete...")
