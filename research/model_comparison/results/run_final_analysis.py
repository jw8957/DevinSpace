import logging
from datetime import datetime
import os
from typing import Dict, Any

class FinalAnalysis:
    def __init__(self, results_dir='../results', figures_dir='../figures'):
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('final_analysis')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            os.path.join(self.results_dir, 'final_analysis.log'))
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger
    
    def run_complete_analysis(self, training_results: Dict[str, Any]):
        """Run all analyses and generate final report"""
        self.logger.info("Starting final analysis pipeline")
        
        try:
            # Import all analysis modules
            from .analyzers import (
                compare_architectures,
                analyze_convergence,
                analyze_error_patterns,
                analyze_content_type_performance,
                analyze_cross_lingual_transfer,
                analyze_semantic_preservation,
                analyze_syntactic_patterns
            )
            from .analyzers.debug_models import ModelDebugger
            from .analyzers.test_scenarios import TestScenarioManager
            from .analyzers.validate_results import validate_and_report
            from .analyzers.visualization import create_dashboard
            from .analyzers.recommendations import generate_recommendations
            
            # Run architecture analysis
            self.logger.info("Analyzing model architectures")
            arch_results = compare_architectures(
                training_results['bilstm_model'],
                training_results['attention_model']
            )
            
            # Run performance analysis
            self.logger.info("Analyzing model performance")
            perf_results = {
                'convergence': analyze_convergence(training_results['metrics']),
                'errors': analyze_error_patterns(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['texts']
                ),
                'content_types': analyze_content_type_performance(
                    training_results['predictions'],
                    training_results['labels'],
                    training_results['texts']
                )
            }
            
            # Run linguistic analysis
            self.logger.info("Analyzing linguistic aspects")
            ling_results = {
                'cross_lingual': analyze_cross_lingual_transfer(
                    training_results['predictions'],
                    training_results['languages'],
                    training_results['labels']
                ),
                'semantic': analyze_semantic_preservation(
                    training_results['predictions'],
                    training_results['texts'],
                    training_results['labels'],
                    training_results['original_texts']
                ),
                'syntax': analyze_syntactic_patterns(
                    training_results['predictions'],
                    training_results['texts'],
                    training_results['labels']
                )
            }
            
            # Run model debugging
            self.logger.info("Running model diagnostics")
            debugger = ModelDebugger()
            debug_results = debugger.generate_debug_report(
                training_results['bilstm_model'],
                training_results['outputs'],
                training_results['attention_weights']
            )
            
            # Run test scenarios
            self.logger.info("Running test scenarios")
            test_manager = TestScenarioManager()
            test_results = test_manager.run_test_cases(
                training_results['bilstm_model'],
                training_results['tokenizer'],
                training_results['device']
            )
            
            # Validate results
            self.logger.info("Validating results")
            validation_status = validate_and_report({
                'architecture': arch_results,
                'performance': perf_results,
                'linguistic': ling_results,
                'debug': debug_results,
                'tests': test_results
            })
            
            # Generate visualizations
            self.logger.info("Creating visualization dashboard")
            create_dashboard()
            
            # Generate recommendations
            self.logger.info("Generating final recommendations")
            recommendations = generate_recommendations(training_results)
            
            # Compile final report
            self.logger.info("Compiling final report")
            final_report = self._compile_final_report(
                arch_results, perf_results, ling_results,
                debug_results, test_results, recommendations
            )
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            raise
    
    def _compile_final_report(self, *args):
        """Compile all results into final report"""
        report = "# Final Model Comparison Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        
        sections = {
            'Architecture Analysis': args[0],
            'Performance Analysis': args[1],
            'Linguistic Analysis': args[2],
            'Model Diagnostics': args[3],
            'Test Results': args[4],
            'Recommendations': args[5]
        }
        
        for title, content in sections.items():
            report += f"## {title}\n\n{content}\n\n"
        
        return report
