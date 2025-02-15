import json
from collections import defaultdict
import statistics

def analyze_jsonl(filename):
    print(f"\nAnalyzing {filename}:")
    lengths = defaultdict(list)
    total = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            orig_len = len(data['origin'].split())
            clean_len = len(data['rephrase_with_img'].split())
            lengths['original'].append(orig_len)
            lengths['cleaned'].append(clean_len)
            total += 1
    
    print(f"Total samples: {total}")
    print("\nLength statistics:")
    for key in ['original', 'cleaned']:
        lens = lengths[key]
        print(f"\n{key.title()}:")
        print(f"  Mean length: {statistics.mean(lens):.1f}")
        print(f"  Median length: {statistics.median(lens):.1f}")
        print(f"  Min length: {min(lens)}")
        print(f"  Max length: {max(lens)}")
    
    print(f"\nAverage reduction ratio: {sum(lengths['cleaned'])/sum(lengths['original']):.2%}")

# Analyze both English and Chinese samples
analyze_jsonl('/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl')
analyze_jsonl('/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl')
