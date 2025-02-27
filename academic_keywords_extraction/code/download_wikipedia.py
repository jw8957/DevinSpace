import os
from datasets import load_dataset
import json

# Create directory for saving data
os.makedirs("english_wikipedia", exist_ok=True)

# Load only the English subset of Wikipedia
print("Loading English Wikipedia dataset...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

print(f"Dataset loaded with {len(dataset)} articles")

# Save a sample of the data to verify content
print("Saving sample data...")
sample_size = 1000
sample = dataset.select(range(sample_size))
sample.to_json("english_wikipedia/sample.json")

print(f"Sample of {sample_size} articles saved to english_wikipedia/sample.json")

# Save the dataset in chunks to avoid memory issues
print("Saving full dataset in chunks...")
chunk_size = 100000  # 100k articles per chunk
num_chunks = (len(dataset) + chunk_size - 1) // chunk_size  # Ceiling division

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(dataset))
    
    print(f"Processing chunk {i+1}/{num_chunks} (articles {start_idx} to {end_idx-1})...")
    
    chunk = dataset.select(range(start_idx, end_idx))
    chunk.to_json(f"english_wikipedia/chunk_{i+1}_of_{num_chunks}.json")
    
    print(f"Chunk {i+1}/{num_chunks} saved")

print("Download complete!")
