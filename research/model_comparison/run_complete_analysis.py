import logging
import os
from results import (
    FinalAnalysis,
    TrainingMonitor,
    validate_and_report,
    create_dashboard
)

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('complete_analysis')
    
    try:
        # Load training results
        logger.info("Loading training results")
        results_dir = os.path.join('results')
        
        # Initialize analysis components
        # For development/testing, use test data
        from results.analyzers.test_data import get_test_data
        test_data = get_test_data()
        
        # Initialize analysis components
        final_analysis = FinalAnalysis(results_dir)
        
        # Run complete analysis pipeline
        logger.info("Running complete analysis pipeline")
        final_report = final_analysis.run_complete_analysis(test_data)
        
        # Save final report
        report_path = os.path.join(results_dir, 'final_report.md')
        with open(report_path, 'w') as f:
            f.write(final_report)
        
        logger.info(f"Final report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()
