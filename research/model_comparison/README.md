# Model Architecture Comparison

This directory contains the analysis and results of comparing two model architectures for content filtering:
1. BiLSTM + Attention
2. Attention-only

## Experiment Setup
- Dataset: Combined English (1991 samples) and Chinese (1999 samples)
- Training split: 70%
- Validation split: 15%
- Test split: 15%
- Base model: sentence-transformers/all-MiniLM-L6-v2
- Learning rate: 2e-5
- Batch size: 16
- Early stopping patience: 3

## Directory Structure
- `figures/`: Visualization plots comparing model performance
- `results/`: Raw metrics and evaluation results

## Results
Results will be updated once the experiments complete.

### Current Progress
- Training BiLSTM model (Epoch 1/5)
- Next: Training Attention-only model
- Final metrics and analysis pending

## Analysis
Analysis will be added after collecting all metrics.
