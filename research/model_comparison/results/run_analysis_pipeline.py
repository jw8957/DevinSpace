import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    def __init__(self, results_dir='../results', figures_dir='../figures'):
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.analysis_modules = [
            ('convergence', 'analyze_convergence'),
            ('errors', 'analyze_errors'),
            ('attention', 'analyze_attention'),
            ('content_types', 'analyze_content_types'),
            ('complexity', 'analyze_complexity'),
            ('cross_lingual', 'analyze_cross_lingual'),
            ('domain_transfer', 'analyze_domain_transfer'),
            ('edge_cases', 'analyze_edge_cases'),
            ('html_context', 'analyze_html_context'),
            ('readability', 'analyze_readability'),
            ('robustness', 'analyze_robustness'),
            ('segmentation', 'analyze_segmentation'),
            ('semantic_preservation', 'analyze_semantic_preservation'),
            ('syntax', 'analyze_syntax'),
            ('text_style', 'analyze_text_style')
        ]
    
    def run_analysis(self, training_results):
        """Run all analysis modules on training results"""
        analysis_results = {}
        
        for module_name, analysis_func in self.analysis_modules:
            logger.info(f"Running {module_name} analysis...")
            try:
                module = __import__(f'analyze_{module_name}')
                analysis_func = getattr(module, analysis_func)
                results = analysis_func(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['texts']
                )
                analysis_results[module_name] = results
                logger.info(f"Completed {module_name} analysis")
            except Exception as e:
                logger.error(f"Error in {module_name} analysis: {str(e)}")
                analysis_results[module_name] = f"Analysis failed: {str(e)}"
        
        return analysis_results
    
    def validate_results(self, analysis_results):
        """Validate analysis results"""
        from validate_results import validate_and_report
        validation_status = validate_and_report(analysis_results)
        return validation_status
    
    def generate_report(self, analysis_results, validation_status):
        """Generate comprehensive report"""
        from generate_comprehensive_report import generate_report
        generate_report(analysis_results)
    
    def run_pipeline(self, training_results):
        """Run complete analysis pipeline"""
        logger.info("Starting analysis pipeline...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Run all analyses
        analysis_results = self.run_analysis(training_results)
        
        # Validate results
        validation_status = self.validate_results(analysis_results)
        
        # Generate report if validation passes
        if all(status == 'complete' 
               for status in validation_status.values()):
            self.generate_report(analysis_results, validation_status)
            logger.info("Analysis pipeline completed successfully")
        else:
            logger.warning("Some analyses incomplete or invalid. "
                         "Check validation report for details.")
        
        return analysis_results, validation_status
