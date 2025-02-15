import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import spacy

def load_nlp_models():
    """Load spaCy models for both languages"""
    return {
        'en': spacy.load('en_core_web_sm'),
        'zh': spacy.load('zh_core_web_sm')
    }

def analyze_sentence_structure(text, nlp):
    """Analyze syntactic structure of text"""
    doc = nlp(text)
    return {
        'sentence_length': len(doc),
        'dep_tree_depth': max(token.head.i - token.i for token in doc),
        'clause_count': len([token for token in doc if token.dep_ == 'ROOT']),
        'noun_phrases': len(list(doc.noun_chunks)),
        'verb_phrases': len([token for token in doc if token.pos_ == 'VERB'])
    }

def analyze_syntactic_patterns(predictions, texts, labels, languages):
    """Analyze model performance across different syntactic patterns"""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')
    
    syntax_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, text, label, lang in zip(predictions, texts, labels, languages):
        try:
            if lang != 'en':  # Skip non-English text for now
                continue
                
            doc = nlp(text)
            
            # Analyze sentence structure
            structure = analyze_sentence_structure(text, nlp)
            
            # Categorize complexity
            complexity = 'simple'
            if structure['dep_tree_depth'] > 5:
                complexity = 'complex'
            elif structure['clause_count'] > 2:
                complexity = 'moderate'
            
            syntax_metrics[complexity]['total'] += 1
            if pred == label:
                syntax_metrics[complexity]['correct'] += 1
                
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            continue
    
    # Calculate accuracy per category
    accuracies = {
        category: metrics['correct'] / metrics['total']
        for category, metrics in syntax_metrics.items()
        if metrics['total'] > 0
    }
    
    return accuracies, syntax_metrics
            if structure['dep_tree_depth'] > 5:
                complexity = 'complex'
            elif structure['clause_count'] > 2:
                complexity = 'moderate'
            
            syntax_metrics[complexity]['total'] += 1
            if pred == label:
                syntax_metrics[complexity]['correct'] += 1
                
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            continue
    
    # Calculate accuracy per category
    accuracies = {
        category: metrics['correct'] / metrics['total']
        for category, metrics in syntax_metrics.items()
        if metrics['total'] > 0
    }
    
    return accuracies, syntax_metrics

def plot_syntax_analysis(bilstm_acc, attn_acc, syntax_metrics):
    """Plot syntactic analysis results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance by complexity
    categories = list(bilstm_acc.keys())
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, [bilstm_acc[c] for c in categories], width, 
            label='BiLSTM+Attention')
    ax1.bar(x + width/2, [attn_acc[c] for c in categories], width, 
            label='Attention-only')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance by Syntactic Complexity')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.title() for c in categories])
    ax1.legend()
    
    # Distribution of syntactic patterns
    total = sum(metrics['total'] for metrics in syntax_metrics.values())
    pattern_dist = {k: v['total']/total*100 
                   for k, v in syntax_metrics.items()}
    
    ax2.pie(pattern_dist.values(), 
            labels=[k.title() for k in pattern_dist.keys()], 
            autopct='%1.1f%%')
    ax2.set_title('Distribution of Syntactic Patterns')
    
    plt.tight_layout()
    plt.savefig('../figures/syntax_analysis.png')
    plt.close()
