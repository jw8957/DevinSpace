from datasets import load_dataset
import json
import os

# Create directory for the dataset
os.makedirs('armanc_scientific', exist_ok=True)

# Function to save a sample of a dataset
def save_sample(dataset_name, config_name, sample_size=1000, output_dir=None):
    if output_dir is None:
        output_dir = dataset_name.replace('/', '_')
    
    print(f'Loading {dataset_name} dataset with config {config_name}...')
    try:
        # Try to load the dataset with the specified config
        ds = load_dataset(dataset_name, config_name, split='train', trust_remote_code=True)
        
        # Take a sample
        sample = ds.select(range(min(sample_size, len(ds))))
        
        # Save the sample
        output_file = f'{output_dir}/{config_name}_sample.json'
        sample.to_json(output_file)
        
        print(f'Saved sample of {len(sample)} examples to {output_file}')
        return True
    except Exception as e:
        print(f'Error loading {dataset_name} with config {config_name}: {e}')
        return False

# Load and save samples from armanc/scientific_papers with both configs
dataset_name = 'armanc/scientific_papers'
output_dir = 'armanc_scientific'

for config in ['pubmed', 'arxiv']:
    print(f'\nProcessing {dataset_name} with config {config}...')
    success = save_sample(dataset_name, config, sample_size=1000, output_dir=output_dir)
    if success:
        print(f'Successfully processed {dataset_name} with config {config}')
    else:
        print(f'Failed to process {dataset_name} with config {config}')
