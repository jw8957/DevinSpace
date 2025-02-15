import json
import re
from collections import defaultdict

def find_common_patterns(filename, num_samples=100):
    print(f"\nAnalyzing patterns in {filename}:")
    
    # Common patterns to look for
    patterns = {
        'navigation': r'(home|menu|navigation|breadcrumb)',
        'social_media': r'(facebook|twitter|instagram|share|follow)',
        'ads': r'(advertisement|sponsored|promotion)',
        'related': r'(related|recommended|you might also like)',
        'comments': r'(comments|replies|discuss)',
        'login': r'(login|sign in|register|forgot password)',
    }
    
    pattern_counts = defaultdict(int)
    removed_text_samples = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:num_samples]
    
    for sample in samples:
        orig = sample['origin']
        clean = sample['rephrase_with_img']
        
        # Find removed text by comparing original and cleaned
        orig_lines = set(orig.split('\n'))
        clean_lines = set(clean.split('\n'))
        removed_lines = orig_lines - clean_lines
        
        # Check patterns in removed lines
        for line in removed_lines:
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, line.lower()):
                    pattern_counts[pattern_name] += 1
                    if len(removed_text_samples) < 5:  # Store up to 5 examples
                        removed_text_samples.append((pattern_name, line.strip()))
    
    # Print statistics
    print("\nCommon patterns found in removed text:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{pattern}: {count} occurrences")
    
    print("\nExample removed text:")
    for pattern, text in removed_text_samples:
        print(f"\nPattern: {pattern}")
        print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")

# Analyze both files
find_common_patterns('/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl')
find_common_patterns('/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl')
