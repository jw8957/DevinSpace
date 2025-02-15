import json
import matplotlib.pyplot as plt
from datetime import datetime

def combine_analysis_results(results_dict):
    """Combine all analysis results into a comprehensive report"""
    report = f"""# Comprehensive Model Architecture Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Overall Performance Metrics
### 1.1 Basic Metrics
- Accuracy: BiLSTM ({results_dict['accuracy']['bilstm']:.4f}) vs Attention-only ({results_dict['accuracy']['attention']:.4f})
- Training Time: BiLSTM ({results_dict['training_time']['bilstm']/60:.2f} min) vs Attention-only ({results_dict['training_time']['attention']/60:.2f} min)
- Memory Usage: BiLSTM ({results_dict['memory']['bilstm']/1e6:.2f} MB) vs Attention-only ({results_dict['memory']['attention']/1e6:.2f} MB)
- Inference Latency: BiLSTM ({results_dict['latency']['bilstm']*1000:.2f} ms) vs Attention-only ({results_dict['latency']['attention']*1000:.2f} ms)

### 1.2 Convergence Analysis
{results_dict.get('convergence_analysis', 'Convergence analysis results pending...')}

## 2. Language-Specific Performance
### 2.1 Cross-lingual Transfer
{results_dict.get('cross_lingual_analysis', 'Cross-lingual analysis results pending...')}

### 2.2 Language-specific Metrics
{results_dict.get('language_metrics', 'Language-specific metrics pending...')}

## 3. Content Analysis
### 3.1 Content Type Performance
{results_dict.get('content_type_analysis', 'Content type analysis results pending...')}

### 3.2 Text Complexity
{results_dict.get('complexity_analysis', 'Complexity analysis results pending...')}

### 3.3 Edge Cases
{results_dict.get('edge_case_analysis', 'Edge case analysis results pending...')}

## 4. Robustness Analysis
### 4.1 Input Variations
{results_dict.get('robustness_analysis', 'Robustness analysis results pending...')}

### 4.2 Semantic Preservation
{results_dict.get('semantic_analysis', 'Semantic preservation analysis pending...')}

## 5. Structural Analysis
### 5.1 HTML Context Performance
{results_dict.get('html_context_analysis', 'HTML context analysis pending...')}

### 5.2 Text Segmentation
{results_dict.get('segmentation_analysis', 'Segmentation analysis pending...')}

## 6. Readability and Style
### 6.1 Readability Levels
{results_dict.get('readability_analysis', 'Readability analysis pending...')}

### 6.2 Text Style Performance
{results_dict.get('style_analysis', 'Style analysis pending...')}

## 7. Recommendations
Based on the comprehensive analysis above, here are the key recommendations:
{results_dict.get('recommendations', 'Recommendations pending completion of all analyses...')}

## 8. Visualizations
All visualization plots can be found in the '../figures/' directory:
- convergence_analysis.png
- language_performance.png
- content_type_analysis.png
- complexity_analysis.png
- edge_case_analysis.png
- robustness_analysis.png
- semantic_preservation.png
- html_context_analysis.png
- segmentation_analysis.png
- readability_analysis.png
- style_analysis.png
"""
    return report

def save_comprehensive_report(report_content, output_path):
    """Save the comprehensive report to a file"""
    with open(output_path, 'w') as f:
        f.write(report_content)

def generate_report(results_dict, output_path='../comprehensive_report.md'):
    """Generate and save the comprehensive report"""
    report_content = combine_analysis_results(results_dict)
    save_comprehensive_report(report_content, output_path)
    print(f"Comprehensive report saved to {output_path}")
