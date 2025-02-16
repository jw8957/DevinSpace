import json
import logging
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_test_data(input_files: list, output_file: str, num_samples: int = 5):
    """Prepare a balanced test dataset from multiple input files."""
    logger.info(f"Preparing test data from {len(input_files)} files")
    samples = []
    
    # Read samples from each file
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                file_samples = []
                for line in f:
                    try:
                        sample = json.loads(line)
                        file_samples.append(sample)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {file_path}")
                        continue
                
                # Take a smaller portion of samples from each file
                if file_samples:
                    selected = random.sample(file_samples, min(num_samples, len(file_samples)))
                    samples.extend(selected)
                    logger.info(f"Selected {len(selected)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            continue
    
    if not samples:
        raise ValueError("No samples could be loaded from input files")
    
    # Shuffle samples
    random.shuffle(samples)
    
    # Write test dataset
    with open(output_file, 'w') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Created test dataset with {len(samples)} samples at {output_file}")

if __name__ == '__main__':
    # Prepare test data from both English and Chinese files
    input_files = [
        'RephraseModel/data/train.json',  # English samples
        'RephraseModel/data/test.json'    # Chinese samples
    ]
    prepare_test_data(input_files, '/tmp/test_data.json')
