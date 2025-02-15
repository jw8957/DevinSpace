import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

def create_dashboard(figures_dir='../figures', output_path='../dashboard.png'):
    """Create a single dashboard combining all visualization plots"""
    # List of expected visualization files
    plot_files = [
        'training_progress.png',
        'convergence_analysis.png',
        'language_performance.png',
        'content_type_analysis.png',
        'complexity_analysis.png',
        'edge_case_analysis.png',
        'robustness_analysis.png',
        'semantic_preservation.png',
        'html_context_analysis.png',
        'segmentation_analysis.png',
        'readability_analysis.png',
        'style_analysis.png'
    ]
    
    # Load available plots
    plots = []
    for filename in plot_files:
        path = os.path.join(figures_dir, filename)
        if os.path.exists(path):
            try:
                img = Image.open(path)
                plots.append((filename.replace('.png', '').replace('_', ' ').title(), 
                            img))
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    if not plots:
        print("No visualization plots found")
        return
    
    # Calculate grid dimensions
    n_plots = len(plots)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 6*n_rows))
    fig.suptitle('Model Comparison Analysis Dashboard', 
                 fontsize=16, y=0.98)
    
    for idx, (title, img) in enumerate(plots):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def update_dashboard(results_dir='../results', figures_dir='../figures'):
    """Update dashboard with latest visualizations"""
    # First ensure all individual plots are up to date
    from process_metrics import plot_training_progress
    from analyze_convergence import plot_convergence_analysis
    from analyze_language_performance import plot_language_comparison
    from analyze_content_types import plot_content_type_analysis
    from analyze_complexity import plot_complexity_analysis
    from analyze_edge_cases import plot_edge_case_analysis
    from analyze_robustness import plot_robustness_analysis
    from analyze_semantic_preservation import plot_semantic_preservation
    from analyze_html_context import plot_html_context_analysis
    from analyze_segmentation import plot_segmentation_analysis
    from analyze_readability import plot_readability_analysis
    from analyze_text_style import plot_style_analysis
    
    # Then create dashboard
    create_dashboard(figures_dir)
