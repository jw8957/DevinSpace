import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def identify_content_type(text: str) -> List[str]:
    """Identify types of content in a text segment"""
    patterns = {
        'breadcrumb': [
            r'(home|category|products?)\s*[>/]',
            r'>\s*(you are here|navigation)',
            r'breadcrumb'
        ],
        'navigation': [
            r'(main|site)\s*navigation',
            r'(menu|nav)\s*(items?|links?)',
            r'skip\s*to\s*(main\s*)?content'
        ],
        'advertisement': [
            r'sponsored\s*(content|post|link)',
            r'advertisement',
            r'promoted\s*(content|stories)',
            r'ads?\s*by\s*google'
        ],
        'social_media': [
            r'share\s*(this|on)',
            r'follow\s*us',
            r'like\s*(us|this)',
            r'social\s*media'
        ],
        'metadata': [
            r'posted\s*(on|by)',
            r'author:\s*',
            r'date:\s*',
            r'comments?\s*\(\d+\)',
            r'tags?:\s*'
        ],
        'related_content': [
            r'related\s*(articles?|posts?|stories?)',
            r'you\s*might\s*(also\s*)?like',
            r'recommended\s*for\s*you',
            r'popular\s*(posts?|articles?)'
        ]
    }
    
    content_types = []
    for content_type, pattern_list in patterns.items():
        if any(re.search(pattern, text.lower()) for pattern in pattern_list):
            content_types.append(content_type)
    return content_types

def analyze_file_content_types(file_path: str) -> Dict:
    """Analyze content types in a file"""
    logger.info(f"Analyzing content types in {file_path}")
    content_type_stats = Counter()
    total_segments = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            orig_text = data['origin']
            clean_text = data['rephrase_with_img']
            
            # Find removed segments
            orig_segments = orig_text.split('\n')
            clean_segments = clean_text.split('\n')
            
            # Use set difference to find removed content
            removed_segments = set(orig_segments) - set(clean_segments)
            
            for segment in removed_segments:
                if segment.strip():
                    content_types = identify_content_type(segment)
                    if content_types:
                        content_type_stats.update(content_types)
                    else:
                        content_type_stats['other_boilerplate'] += 1
                    total_segments += 1
    
    return {
        'stats': dict(content_type_stats),
        'total_segments': total_segments
    }

def plot_content_type_distribution(en_stats: Dict, zh_stats: Dict):
    """Create visualization of content type distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # English distribution
    en_labels = list(en_stats['stats'].keys())
    en_values = [en_stats['stats'][k] / en_stats['total_segments'] * 100 for k in en_labels]
    sns.barplot(x=en_values, y=en_labels, ax=ax1)
    ax1.set_title('English Content Type Distribution')
    ax1.set_xlabel('Percentage of Filtered Content')
    
    # Chinese distribution
    zh_labels = list(zh_stats['stats'].keys())
    zh_values = [zh_stats['stats'][k] / zh_stats['total_segments'] * 100 for k in zh_labels]
    sns.barplot(x=zh_values, y=zh_labels, ax=ax2)
    ax2.set_title('Chinese Content Type Distribution')
    ax2.set_xlabel('Percentage of Filtered Content')
    
    plt.tight_layout()
    plt.savefig('content_type_distribution.png')
    plt.close()

if __name__ == "__main__":
    en_file = '/home/ubuntu/attachments/f0b4f54e-a7fc-49a6-9b0b-9b343acacf4b/cc_sample.20250215_en_2000.rephrase.jsonl'
    zh_file = '/home/ubuntu/attachments/29ad0211-497f-4350-aedd-335fd3e3d4fc/cc_sample.20250215_zh-hans_2000.rephrase.jsonl'
    
    # Analyze content types
    en_stats = analyze_file_content_types(en_file)
    zh_stats = analyze_file_content_types(zh_file)
    
    # Plot distribution
    plot_content_type_distribution(en_stats, zh_stats)
