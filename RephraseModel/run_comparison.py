from compare_architectures import compare_models
from visualize_comparison import plot_comparison_results
from data_processor import ContentDataset
from torch.utils.data import DataLoader, random_split
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_datasets():
    """Prepare datasets for training and evaluation"""
    # Load datasets
    en_file = "/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl"
    zh_file = "/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl"
    
    en_dataset = ContentDataset(en_file)
    zh_dataset = ContentDataset(zh_file)
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([en_dataset, zh_dataset])
    total_size = len(combined_dataset)
    
    # Split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    return train_loader, val_loader, test_loader

def main():
    """Run experiments and generate report"""
    logger.info("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets()
    
    logger.info("Running model comparison experiments...")
    results = compare_models(train_loader, val_loader, test_loader)
    
    logger.info("Generating visualizations...")
    plot_comparison_results(results)
    
    logger.info("Generating comparison report...")
    with open('architecture_comparison_report.md', 'w') as f:
        f.write(f"""# Architecture Comparison Report

## Performance Metrics

1. Accuracy
   - BiLSTM+Attention: {results['accuracy']['bilstm']:.4f}
   - Attention-only: {results['accuracy']['attention']:.4f}

2. Inference Latency (ms)
   - BiLSTM+Attention: {results['latency']['bilstm']*1000:.2f}
   - Attention-only: {results['latency']['attention']*1000:.2f}

3. Memory Usage (MB)
   - BiLSTM+Attention: {results['memory']['bilstm']/1e6:.2f}
   - Attention-only: {results['memory']['attention']/1e6:.2f}

4. Training Time (minutes)
   - BiLSTM+Attention: {results['training_time']['bilstm']/60:.2f}
   - Attention-only: {results['training_time']['attention']/60:.2f}

## Analysis
### Performance Analysis
1. Accuracy Comparison
   - Relative difference: {((results['accuracy']['attention'] - results['accuracy']['bilstm'])/results['accuracy']['bilstm']*100):.2f}%
   - Impact on content filtering quality
   
2. Computational Efficiency
   - Latency improvement: {((results['latency']['bilstm'] - results['latency']['attention'])/results['latency']['bilstm']*100):.2f}%
   - Memory savings: {((results['memory']['bilstm'] - results['memory']['attention'])/results['memory']['bilstm']*100):.2f}%
   - Training time reduction: {((results['training_time']['bilstm'] - results['training_time']['attention'])/results['training_time']['bilstm']*100):.2f}%

### Trade-offs
1. Model Complexity
   - BiLSTM+Attention: More parameters, potentially better context understanding
   - Attention-only: Simpler architecture, focus on attention mechanisms

2. Resource Requirements
   - Training resources
   - Inference requirements
   - Deployment considerations

## Recommendations
[Will be updated based on experimental results]

## Visualizations
See 'architecture_comparison.png' for detailed performance comparisons.
""")
    
    logger.info("Experiments and report generation completed")

if __name__ == "__main__":
    main()
