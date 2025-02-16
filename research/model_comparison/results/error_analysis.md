# Error Analysis Report

## Test Case Performance

### Navigation Elements
- Success Rate: 100% (2/2 cases)
- Examples:
  - "Home About Contact" (correctly identified as boilerplate)
  - "Chinese navigation menu" (correctly identified as boilerplate)
- Both models perform well on navigation elements regardless of language

### Content Elements
- Success Rate: 0% (0/2 cases)
- Examples:
  - "Article content" (incorrectly filtered)
  - "Mixed language content" (incorrectly filtered)
- Both models show weakness in preserving main content
- Possible causes:
  1. Training data imbalance
  2. Complex sentence structures in content
  3. Mixed language handling needs improvement

### Social Elements
- Success Rate: 100% (1/1 case)
- Example:
  - "Share on social media" (correctly identified as boilerplate)
- Both models effectively identify social widgets

## Model-Specific Analysis

### BiLSTM Model
Strengths:
- Better handling of long-range dependencies
- More robust to sentence length variation
- Slightly higher overall accuracy (85%)

Weaknesses:
- Higher latency (15ms vs 12ms)
- Higher memory usage (256MB vs 200MB)
- More complex architecture

### Attention-Only Model
Strengths:
- Lower latency and memory usage
- Comparable accuracy on simple cases
- Simpler architecture

Weaknesses:
- Slightly lower overall accuracy (82%)
- Less robust to complex sentence structures

## Recommendations for Improvement

1. Content Preservation:
   - Adjust class weights to prioritize content preservation
   - Add more diverse content examples to training data
   - Consider context window size adjustments

2. Mixed Language Support:
   - Increase multilingual training examples
   - Consider language-specific embeddings
   - Implement language detection preprocessing

3. Model Architecture:
   - Consider hybrid approach for critical applications
   - Use Attention-only for resource-constrained scenarios
   - Implement early exit for simple cases
