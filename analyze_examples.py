import json
import torch
from data_processor import ContentDataset
from model import ContentFilterModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_filtering_examples(model_path, data_file, num_examples=5):
    """Analyze specific examples of content filtering decisions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ContentFilterModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = ContentDataset(data_file)
    
    examples = []
    with torch.no_grad():
        for i in range(min(len(dataset), num_examples * 10)):  # Sample more to find interesting cases
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)
            
            # Get text from tokenizer
            text = dataset.tokenizer.decode(input_ids[0][attention_mask[0] == 1])
            
            # Get predictions and labels for non-padded tokens
            valid_preds = preds[0][attention_mask[0] == 1].cpu().numpy()
            valid_labels = labels[attention_mask[0] == 1].numpy()
            
            # Check if this is an interesting example (has both kept and filtered content)
            if len(valid_preds) > 0 and (valid_preds != valid_labels).any():
                examples.append({
                    'text': text,
                    'predictions': valid_preds.tolist(),
                    'labels': valid_labels.tolist(),
                    'analysis': analyze_filtering_decision(text, valid_preds, valid_labels)
                })
                
                if len(examples) >= num_examples:
                    break
    
    return examples

def analyze_filtering_decision(text, predictions, labels):
    """Analyze why certain content was filtered"""
    analysis = []
    
    # Common patterns for different types of content
    patterns = {
        'breadcrumb': ['>', '/', 'Home', 'Category'],
        'navigation': ['Menu', 'Navigation', 'Skip to content'],
        'advertisement': ['Ad', 'Advertisement', 'Sponsored'],
        'social': ['Share', 'Follow', 'Like', 'Tweet'],
        'metadata': ['Posted on', 'Author:', 'Comments', 'Tags:']
    }
    
    # Analyze each filtered segment
    current_segment = {'text': '', 'pred': None, 'label': None}
    segments = []
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if i > 0 and (pred != predictions[i-1] or label != labels[i-1]):
            if current_segment['text']:
                segments.append(current_segment)
            current_segment = {'text': '', 'pred': pred, 'label': label}
        current_segment['text'] += text[i] if i < len(text) else ''
    
    if current_segment['text']:
        segments.append(current_segment)
    
    # Analyze each segment
    for segment in segments:
        if segment['pred'] == 0:  # Model predicted to filter this
            reasons = []
            for content_type, keywords in patterns.items():
                if any(keyword.lower() in segment['text'].lower() for keyword in keywords):
                    reasons.append(content_type)
            
            if reasons:
                analysis.append({
                    'segment': segment['text'].strip(),
                    'predicted': 'filter',
                    'actual': 'keep' if segment['label'] == 1 else 'filter',
                    'likely_reasons': reasons
                })
    
    return analysis

if __name__ == "__main__":
    model_path = 'model_outputs/best_model.pt'
    en_file = '/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl'
    zh_file = '/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl'
    
    print("Waiting for training to complete...")
