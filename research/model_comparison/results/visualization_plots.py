import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_performance_plots(metrics):
    """Create performance comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy Plot
    ax = axes[0, 0]
    data = [metrics['accuracy']['bilstm'], metrics['accuracy']['attention']]
    sns.barplot(x=['BiLSTM+Attention', 'Attention-only'], y=data, ax=ax)
    ax.set_title('Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    
    # Latency Plot
    ax = axes[0, 1]
    data = [metrics['latency']['bilstm']*1000, metrics['latency']['attention']*1000]
    sns.barplot(x=['BiLSTM+Attention', 'Attention-only'], y=data, ax=ax)
    ax.set_title('Latency Comparison')
    ax.set_ylabel('Latency (ms)')
    
    # Memory Usage Plot
    ax = axes[1, 0]
    data = [metrics['memory']['bilstm']/1e6, metrics['memory']['attention']/1e6]
    sns.barplot(x=['BiLSTM+Attention', 'Attention-only'], y=data, ax=ax)
    ax.set_title('Memory Usage Comparison')
    ax.set_ylabel('Memory (MB)')
    
    # Training Time Plot
    ax = axes[1, 1]
    data = [metrics['training_time']['bilstm']/60, metrics['training_time']['attention']/60]
    sns.barplot(x=['BiLSTM+Attention', 'Attention-only'], y=data, ax=ax)
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Time (minutes)')
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png')
    plt.close()

def create_convergence_plots(metrics):
    """Create training convergence plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training Loss
    epochs = range(1, len(metrics['training_loss']['bilstm']) + 1)
    ax1.plot(epochs, metrics['training_loss']['bilstm'], 'b-', label='BiLSTM+Attention')
    ax1.plot(epochs, metrics['training_loss']['attention'], 'r-', label='Attention-only')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Validation Loss
    ax2.plot(epochs, metrics['validation_loss']['bilstm'], 'b-', label='BiLSTM+Attention')
    ax2.plot(epochs, metrics['validation_loss']['attention'], 'r-', label='Attention-only')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/convergence_comparison.png')
    plt.close()

def create_test_scenario_plots(test_results):
    """Create test scenario performance plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall Test Results
    results = ['Passed', 'Failed']
    counts = [test_results['passed'], test_results['failed']]
    sns.barplot(x=results, y=counts, ax=ax1)
    ax1.set_title('Test Scenario Results')
    ax1.set_ylabel('Count')
    
    # Results by Category
    categories = {}
    for case in test_results['cases']:
        cat = case['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'total': 0}
        categories[cat]['total'] += 1
        if case['passed']:
            categories[cat]['passed'] += 1
    
    cat_names = list(categories.keys())
    cat_accuracy = [categories[cat]['passed']/categories[cat]['total'] for cat in cat_names]
    sns.barplot(x=cat_names, y=cat_accuracy, ax=ax2)
    ax2.set_title('Performance by Category')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('figures/test_scenario_results.png')
    plt.close()

if __name__ == '__main__':
    # Load test data
    from analyzers.test_data import get_test_data
    test_data = get_test_data()
    
    # Create plots
    create_performance_plots(test_data['metrics'])
    create_convergence_plots(test_data['metrics'])
    create_test_scenario_plots(test_data['test_results'])
