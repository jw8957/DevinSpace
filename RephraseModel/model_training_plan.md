# NLP Model Training Plan for Content Filtering

## 1. Task Definition
- **Type**: Sequence Tagging/Classification
- **Input**: Text content split into sentences
- **Output**: Binary labels for each sentence (keep/remove)
- **Base Model**: sentence-transformers/all-MiniLM-L6-v2

## 2. Data Preparation
### 2.1 Dataset Creation
- Extract sentence pairs from origin/rephrase_with_img
- Create labels by aligning original and cleaned content
- Split data: 80% training, 10% validation, 10% test
- Balance dataset to handle class imbalance (~90% keep rate observed)

### 2.2 Preprocessing
- Sentence segmentation using spaCy/NLTK
- Handle both English and Chinese content
- Preserve markdown formatting and image tags
- Create attention masks for special tokens

## 3. Model Architecture
### 3.1 Base Model (all-MiniLM-L6-v2)
- Lightweight transformer architecture
- Strong semantic understanding
- Multilingual support
- 384-dimensional embeddings

### 3.2 Classification Head
- Sentence embedding layer from base model
- Additional layers:
  - Bidirectional LSTM for context
  - Attention mechanism for neighboring sentences
  - Binary classification layer

## 4. Training Strategy
### 4.1 Loss Function
- Binary Cross-Entropy
- Optional: Add class weights to handle imbalance
- Consider contextual loss component

### 4.2 Training Process
- Fine-tune in stages:
  1. Freeze base model, train classifier
  2. Gradually unfreeze layers
- Use early stopping and model checkpoints
- Batch size: 32-64 sentences
- Learning rate: 2e-5 with warmup

## 5. Evaluation Metrics
- Precision/Recall for both classes
- F1-Score
- Compare against rule-based baseline
- Manual evaluation of edge cases

## 6. Inference Pipeline
1. Split input text into sentences
2. Generate embeddings using base model
3. Apply classification head
4. Post-process to maintain document structure
5. Reconstruct filtered content

## 7. Considerations
- Handle both English and Chinese content
- Preserve document structure and formatting
- Consider sentence context in decisions
- Balance between precision and recall
- Maintain image tags and markdown syntax

## 8. Implementation Phases
1. Data preprocessing and dataset creation
2. Model architecture implementation
3. Training pipeline setup
4. Evaluation and metrics tracking
5. Inference pipeline development
6. Integration with existing system
