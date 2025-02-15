import logging
import os
from run_final_analysis import FinalAnalysis
from monitor_training import TrainingMonitor
from validate_results import validate_and_report
from create_visualization_dashboard import create_dashboard

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
        training_monitor = TrainingMonitor(results_dir)
        final_analysis = FinalAnalysis(results_dir)
        
        # Run complete analysis pipeline
        logger.info("Running complete analysis pipeline")
        final_report = final_analysis.run_complete_analysis(
            training_monitor.get_latest_metrics()
        )
        
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
