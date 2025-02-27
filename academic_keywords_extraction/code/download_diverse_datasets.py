from datasets import load_dataset
import json
import os

# Create directories for each dataset
os.makedirs('armanc_scientific', exist_ok=True)
os.makedirs('pubmed_arxiv_abstracts', exist_ok=True)
os.makedirs('processed_pubmed', exist_ok=True)
os.makedirs('scientific_papers_archive', exist_ok=True)

# Function to save a sample of a dataset
def save_sample(dataset_name, sample_size=1000, output_dir=None):
    if output_dir is None:
        output_dir = dataset_name.replace('/', '_')
    
    print(f'Loading {dataset_name} dataset...')
    try:
        # Try to load the dataset
        ds = load_dataset(dataset_name, split='train', streaming=True)
        
        # Take a sample
        sample = []
        for i, example in enumerate(ds):
            if i >= sample_size:
                break
            sample.append(example)
        
        # Save the sample
        output_file = f'{output_dir}/sample.json'
        with open(output_file, 'w') as f:
            json.dump(sample, f)
        
        print(f'Saved sample of {len(sample)} examples to {output_file}')
        return True
    except Exception as e:
        print(f'Error loading {dataset_name}: {e}')
        return False

# Load and save samples from diverse datasets
datasets_to_process = [
    ('armanc/scientific_papers', 'armanc_scientific'),
    ('brainchalov/pubmed_arxiv_abstracts_data', 'pubmed_arxiv_abstracts'),
    ('JYumeko/processed_pubmed_scientific_papers', 'processed_pubmed'),
    ('scillm/scientific_papers-archive', 'scientific_papers_archive')
]

for dataset_name, output_dir in datasets_to_process:
    print(f'\nProcessing {dataset_name}...')
    success = save_sample(dataset_name, sample_size=1000, output_dir=output_dir)
    if success:
        print(f'Successfully processed {dataset_name}')
    else:
        print(f'Failed to process {dataset_name}')
