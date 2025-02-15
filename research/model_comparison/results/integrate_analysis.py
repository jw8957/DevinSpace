import logging
from datetime import datetime
import os

class AnalysisIntegrator:
    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('analysis_integrator')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(
            os.path.join(self.output_dir, 'analysis.log'))
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_integrated_analysis(self, training_results):
        """Run all analyses in the correct order"""
        self.logger.info("Starting integrated analysis pipeline")
        
        try:
            # Import all analysis modules
            from analyze_architectures import compare_architectures
            from analyze_convergence import analyze_convergence
            from analyze_errors import analyze_error_patterns
            from analyze_attention import analyze_attention_patterns
            from analyze_content_types import analyze_content_type_performance
            from analyze_complexity import analyze_complexity
            from analyze_cross_lingual import analyze_cross_lingual_transfer
            from analyze_domain_transfer import analyze_domain_performance
            from analyze_edge_cases import analyze_edge_case_performance
            from analyze_html_context import analyze_html_context_performance
            from analyze_readability import analyze_readability_performance
            from analyze_robustness import analyze_robustness
            from analyze_segmentation import analyze_segmentation_performance
            from analyze_semantic_preservation import analyze_semantic_preservation
            from analyze_syntax import analyze_syntactic_patterns
            from analyze_text_style import analyze_style_performance
            
            # Run analyses
            self.logger.info("Running architecture analysis")
            arch_results = compare_architectures(
                training_results['bilstm_model'],
                training_results['attention_model']
            )
            
            self.logger.info("Running performance analyses")
            performance_results = {
                'convergence': analyze_convergence(training_results['metrics']),
                'errors': analyze_error_patterns(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['texts']
                ),
                'attention': analyze_attention_patterns(
                    training_results['attention_weights']
                )
            }
            
            self.logger.info("Running content analyses")
            content_results = {
                'content_types': analyze_content_type_performance(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['texts']
                ),
                'complexity': analyze_complexity(
                    training_results['texts']
                ),
                'cross_lingual': analyze_cross_lingual_transfer(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['languages']
                )
            }
            
            # Combine all results
            all_results = {
                'architecture': arch_results,
                'performance': performance_results,
                'content': content_results
            }
            
            self.logger.info("Analysis pipeline completed successfully")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
