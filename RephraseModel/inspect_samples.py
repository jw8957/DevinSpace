import json
import random

def print_sample(filename, num_samples=3):
    print(f"\nExamining {filename}:")
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    selected = random.sample(samples, num_samples)
    for i, sample in enumerate(selected, 1):
        print(f"\nSample {i}:")
        print("\nORIGINAL TEXT:")
        print("-" * 80)
        print(sample['origin'][:500] + "..." if len(sample['origin']) > 500 else sample['origin'])
        print("-" * 80)
        print("\nCLEANED TEXT:")
        print("-" * 80)
        print(sample['rephrase_with_img'][:500] + "..." if len(sample['rephrase_with_img']) > 500 else sample['rephrase_with_img'])
        print("-" * 80)
        print("\n")

# Set random seed for reproducibility
random.seed(42)

# Examine both English and Chinese samples
print_sample('/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl')
print_sample('/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl')
