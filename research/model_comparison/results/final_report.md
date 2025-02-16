# Final Model Comparison Report

Generated: 2025-02-16T08:52:18.471955

## Architecture Analysis

# Model Architecture Comparison

## BiLSTM+Attention Architecture

### Parameters
- Total trainable parameters: 2,167,170

### Layer Composition
- LSTM layers: 1
- LINEAR layers: 2

### Activation Functions
- RELU: 1

## Attention-only Architecture

### Parameters
- Total trainable parameters: 2,461,058

### Layer Composition
- ATTENTION layers: 1
- LINEAR layers: 3

### Activation Functions
- RELU: 1

## Architecture Comparison Summary
- Parameter Ratio: 0.88x
- Key Differences:
  - Sequence Processing: BiLSTM uses bidirectional LSTM for sequential processing
  - Attention Mechanism: Both use attention, but with different architectures
  - Parameter Efficiency: Compare parameter counts and distribution


## Performance Analysis

{'convergence': {'bilstm': {'epochs_to_best': 5, 'best_val_loss': 0.8, 'loss_stability': np.float64(0.20548046676563256), 'final_train_loss': 0.7, 'generalization_gap': -0.10000000000000009}, 'attention': {'epochs_to_best': 5, 'best_val_loss': 0.7, 'loss_stability': np.float64(0.20548046676563253), 'final_train_loss': 0.6, 'generalization_gap': -0.09999999999999998}}, 'errors': [{'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}, {'text': 'Home About Contact', 'predicted': 0, 'actual': 1}, {'text': 'Share on social media', 'predicted': 1, 'actual': 0}], 'content_types': ({'other': 0.5, 'social': 0.0}, {'other': 20, 'social': 10})}

## Linguistic Analysis

{'cross_lingual': {'en': {'correct': 10, 'total': 20}, 'zh': {'correct': 0, 'total': 10}, 'overall': {'correct': 10, 'total': 30}}, 'semantic': {'correct_predictions': [], 'incorrect_predictions': [np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962), np.float32(0.08929968), np.float32(0.1184721), np.float32(0.048918962)]}, 'syntax': ({'simple': 0.5}, defaultdict(<function analyze_syntactic_patterns.<locals>.<lambda> at 0x7ff2a0101620>, {'simple': {'correct': 10, 'total': 20}}))}

## Model Diagnostics

# Model Debug Report

## Gradient Statistics

## Activation Statistics

### hidden_states
- mean: 0.0082
- std: 1.0026
- dead_neuron_percentage: 49.6094
- saturation_percentage: 16.1458

### attention_weights
- mean: -0.0384
- std: 1.0166
- dead_neuron_percentage: 51.1111
- saturation_percentage: 15.1111

## Attention Statistics
- entropy: 2.9849
- sparsity: 0.5744
- coverage: 0.4256


## Test Results

{'timestamp': '2025-02-16T08:52:18.442784', 'total_cases': 5, 'passed': 3, 'failed': 2, 'cases': [{'name': 'navigation_menu', 'passed': True, 'expected': False, 'predicted': False, 'category': 'navigation', 'language': 'en'}, {'name': 'article_content', 'passed': False, 'expected': True, 'predicted': False, 'category': 'content', 'language': 'en'}, {'name': 'social_widgets', 'passed': True, 'expected': False, 'predicted': False, 'category': 'social', 'language': 'en'}, {'name': 'chinese_navigation', 'passed': True, 'expected': False, 'predicted': False, 'category': 'navigation', 'language': 'zh'}, {'name': 'mixed_language', 'passed': False, 'expected': True, 'predicted': False, 'category': 'content', 'language': 'mixed'}]}

## Recommendations

# Model Architecture Recommendations

## Primary Recommendation
Use Attention-only model for better efficiency with comparable accuracy.

## Performance Tradeoffs
- BiLSTM offers higher accuracy but with increased latency. Consider using BiLSTM for non-realtime applications where accuracy is critical.

## Language-specific Considerations

## Deployment Considerations
- Resource Requirements:
  - BiLSTM: 256.0MB memory, 15.0ms latency
  - Attention: 200.0MB memory, 12.0ms latency



