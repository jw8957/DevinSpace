import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from collections import defaultdict

def compute_semantic_similarity(model, text1, text2):
    """Compute semantic similarity between two texts"""
    embeddings = model.encode([text1, text2])
    return 1 - cosine(embeddings[0], embeddings[1])

def analyze_semantic_preservation(predictions, labels, texts, original_texts):
    """Analyze how well models preserve semantic meaning"""
    # Initialize sentence transformer for semantic similarity
    sem_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    preservation_metrics = {
        'correct_predictions': [],
        'incorrect_predictions': []
    }
    
    for pred, label, filtered, original in zip(predictions, labels, texts, original_texts):
        similarity = compute_semantic_similarity(sem_model, filtered, original)
        
        if pred == label:
            preservation_metrics['correct_predictions'].append(similarity)
        else:
            preservation_metrics['incorrect_predictions'].append(similarity)
    
    return preservation_metrics

def plot_semantic_preservation(bilstm_metrics, attn_metrics):
    """Plot semantic preservation analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution of semantic similarity for correct predictions
    sns.boxplot(data=[
        bilstm_metrics['correct_predictions'],
        attn_metrics['correct_predictions']
    ], ax=ax1)
    ax1.set_xticklabels(['BiLSTM+Attention', 'Attention-only'])
    ax1.set_title('Semantic Similarity (Correct Predictions)')
    ax1.set_ylabel('Similarity Score')
    
    # Distribution of semantic similarity for incorrect predictions
    sns.boxplot(data=[
        bilstm_metrics['incorrect_predictions'],
        attn_metrics['incorrect_predictions']
    ], ax=ax2)
    ax2.set_xticklabels(['BiLSTM+Attention', 'Attention-only'])
    ax2.set_title('Semantic Similarity (Incorrect Predictions)')
    ax2.set_ylabel('Similarity Score')
    
    plt.tight_layout()
    plt.savefig('../figures/semantic_preservation.png')
    plt.close()

def analyze_similarity_thresholds(metrics):
    """Analyze performance at different similarity thresholds"""
    thresholds = np.arange(0.5, 1.0, 0.1)
    threshold_metrics = []
    
    for threshold in thresholds:
        correct_above = sum(s >= threshold 
                          for s in metrics['correct_predictions'])
        incorrect_above = sum(s >= threshold 
                            for s in metrics['incorrect_predictions'])
        
        total_correct = len(metrics['correct_predictions'])
        total_incorrect = len(metrics['incorrect_predictions'])
        
        threshold_metrics.append({
            'threshold': threshold,
            'correct_preservation': correct_above / total_correct,
            'incorrect_preservation': incorrect_above / total_incorrect
        })
    
    return threshold_metrics
