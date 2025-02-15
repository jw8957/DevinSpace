import numpy as np
from typing import Dict, Any

def validate_metrics(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """Validate all metrics for consistency and reasonableness"""
    validations = {}
    
    # Accuracy checks
    for model in ['bilstm', 'attention']:
        if 'accuracy' in metrics:
            acc = metrics['accuracy'].get(model, 0)
            validations[f'{model}_accuracy_range'] = 0 <= acc <= 1
    
    # Performance metrics checks
    if 'latency' in metrics:
        validations['latency_positive'] = all(v > 0 for v in metrics['latency'].values())
    if 'memory' in metrics:
        validations['memory_positive'] = all(v > 0 for v in metrics['memory'].values())
    if 'training_time' in metrics:
        validations['training_time_positive'] = all(v > 0 for v in metrics['training_time'].values())
    
    # Cross-validation checks
    if all(key in metrics for key in ['training_loss', 'validation_loss']):
        for model in ['bilstm', 'attention']:
            train_loss = np.mean(metrics['training_loss'][model])
            val_loss = np.mean(metrics['validation_loss'][model])
            validations[f'{model}_loss_reasonable'] = 0 <= train_loss <= 10 and 0 <= val_loss <= 10
    
    return validations

def validate_analysis_results(results: Dict[str, Any]) -> Dict[str, str]:
    """Validate all analysis results for completeness and consistency"""
    validation_status = {}
    
    # Required analysis sections
    required_sections = [
        'convergence_analysis',
        'cross_lingual_analysis',
        'language_metrics',
        'content_type_analysis',
        'complexity_analysis',
        'edge_case_analysis',
        'robustness_analysis',
        'semantic_analysis',
        'html_context_analysis',
        'segmentation_analysis',
        'readability_analysis',
        'style_analysis'
    ]
    
    for section in required_sections:
        if section in results:
            if isinstance(results[section], dict) and results[section]:
                validation_status[section] = 'complete'
            else:
                validation_status[section] = 'incomplete'
        else:
            validation_status[section] = 'missing'
    
    # Validate metrics
    metrics_validation = validate_metrics(results)
    validation_status['metrics_validation'] = (
        'valid' if all(metrics_validation.values()) else 'invalid'
    )
    
    return validation_status

def format_validation_report(validation_status: Dict[str, str]) -> str:
    """Format validation results into a readable report"""
    report = "# Analysis Results Validation Report\n\n"
    
    # Metrics validation
    report += "## Metrics Validation\n"
    report += f"Status: {validation_status.get('metrics_validation', 'not checked')}\n\n"
    
    # Analysis sections validation
    report += "## Analysis Sections Status\n"
    for section, status in validation_status.items():
        if section != 'metrics_validation':
            report += f"- {section.replace('_', ' ').title()}: {status}\n"
    
    return report

def validate_and_report(results_dict: Dict[str, Any], output_path: str = '../validation_report.md'):
    """Run validation and save report"""
    validation_status = validate_analysis_results(results_dict)
    report = format_validation_report(validation_status)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return validation_status
