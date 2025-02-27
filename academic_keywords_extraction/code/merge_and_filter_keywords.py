import os
import csv
from collections import Counter

# Define generic terms to filter out
GENERIC_TERMS = [
    'analysis', 'statistics', 'research', 'study', 'method', 'approach', 'theory',
    'model', 'framework', 'concept', 'process', 'system', 'technique', 'structure',
    'function', 'development', 'application', 'implementation', 'evaluation',
    'assessment', 'investigation', 'examination', 'exploration', 'review',
    'analysis and', 'statistics are', 'research on', 'study of', 'method for',
    'approach to', 'theory of', 'model of', 'framework for', 'concept of',
    'process of', 'system for', 'technique for', 'structure of', 'function of',
    'development of', 'application of', 'implementation of', 'evaluation of',
    'assessment of', 'investigation of', 'examination of', 'exploration of', 'review of',
    'analysis performed', 'analysis showed', 'data analysis', 'performed using',
    'using the', 'based on', 'according to', 'related to', 'compared to',
    'due to', 'in order', 'in order to', 'in this', 'in this study',
    'in the', 'of the', 'to the', 'with the', 'for the',
    'from the', 'by the', 'on the', 'at the', 'as the',
    'is the', 'are the', 'was the', 'were the', 'has been',
    'have been', 'had been', 'will be', 'would be', 'could be',
    'should be', 'may be', 'might be', 'must be', 'can be'
]

# Additional filtering rules
def is_generic_term(keyword):
    # Check if keyword is in the generic terms list
    if keyword.lower() in GENERIC_TERMS:
        return True
    
    # Check if keyword starts with generic words
    generic_starters = ['the ', 'a ', 'an ', 'and ', 'or ', 'of ', 'in ', 'on ', 'at ', 'by ', 'for ', 'with ', 'to ']
    if any(keyword.lower().startswith(starter) for starter in generic_starters):
        return True
    
    # Check if keyword is too short (single word)
    if len(keyword.split()) == 1 and len(keyword) < 5:
        return True
    
    # Check if keyword contains only generic words
    words = keyword.lower().split()
    generic_words = ['the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'is', 'are', 'was', 'were']
    if all(word in generic_words for word in words):
        return True
    
    return False

# Load keywords from a CSV file
def load_keywords(file_path):
    keywords = Counter()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    keyword, count = row[0], int(row[1])
                    keywords[keyword] = count
        return keywords
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return Counter()

# Filter keywords to remove generic terms
def filter_keywords(keywords):
    filtered = Counter()
    for keyword, count in keywords.items():
        if not is_generic_term(keyword):
            filtered[keyword] = count
    return filtered

# Save keywords to a CSV file
def save_keywords(keywords, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keyword', 'Frequency'])
        for keyword, count in keywords.most_common():
            writer.writerow([keyword, count])
    print(f"Saved {len(keywords)} keywords to {file_path}")

# Create a text file with the top keywords
def save_top_keywords_text(keywords, file_path, n=100):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Top {n} Academic Keywords for Google Scholar Search\n\n")
        for i, (keyword, count) in enumerate(keywords.most_common(n), 1):
            f.write(f"{i}. {keyword} ({count})\n")
    print(f"Saved top {n} keywords to {file_path}")

# Main function
def main():
    # Create output directory
    os.makedirs('merged_keywords', exist_ok=True)
    
    # Load existing keywords from Wikipedia and ArXiv ML papers
    print("Loading existing keywords...")
    try:
        existing_keywords = load_keywords('/home/ubuntu/wikipedia_data/expanded_keywords_package/expanded_academic_keywords.csv')
        print(f"Loaded {len(existing_keywords)} existing keywords")
    except Exception as e:
        print(f"Error loading existing keywords: {e}")
        existing_keywords = Counter()
    
    # Load new diverse keywords
    print("Loading diverse keywords...")
    diverse_keywords = load_keywords('keywords/diverse_academic_keywords.csv')
    print(f"Loaded {len(diverse_keywords)} diverse keywords")
    
    # Merge keywords
    print("Merging keywords...")
    merged_keywords = Counter()
    merged_keywords.update(existing_keywords)
    merged_keywords.update(diverse_keywords)
    print(f"Total merged keywords: {len(merged_keywords)}")
    
    # Filter keywords
    print("Filtering keywords...")
    filtered_keywords = filter_keywords(merged_keywords)
    print(f"Keywords after filtering: {len(filtered_keywords)}")
    
    # Save filtered keywords
    save_keywords(filtered_keywords, 'merged_keywords/merged_filtered_keywords.csv')
    
    # Save top keywords
    top_1000 = Counter(dict(filtered_keywords.most_common(1000)))
    top_100 = Counter(dict(filtered_keywords.most_common(100)))
    top_50 = Counter(dict(filtered_keywords.most_common(50)))
    
    save_keywords(top_1000, 'merged_keywords/merged_top_1000_keywords.csv')
    save_keywords(top_100, 'merged_keywords/merged_top_100_keywords.csv')
    save_keywords(top_50, 'merged_keywords/merged_top_50_keywords.csv')
    
    save_top_keywords_text(top_100, 'merged_keywords/top_100_keywords.txt')
    save_top_keywords_text(top_50, 'merged_keywords/top_50_keywords.txt')
    
    # Print top 20 keywords
    print("\nTop 20 filtered keywords:")
    for keyword, count in filtered_keywords.most_common(20):
        print(f"  {keyword}: {count}")
    
    # Analyze keyword categories
    print("\nKeyword category analysis:")
    categories = {
        'medicine': ['surgery', 'pathology', 'oncology', 'cardiology', 'neurology', 'immunology', 'radiology'],
        'biology': ['evolution', 'genetics', 'molecular biology', 'cell biology', 'microbiology', 'ecology'],
        'physics': ['quantum mechanics', 'relativity', 'particle physics', 'astrophysics', 'thermodynamics'],
        'computer science': ['machine learning', 'artificial intelligence', 'neural network', 'deep learning'],
        'mathematics': ['probability', 'statistics', 'algebra', 'geometry', 'calculus'],
        'social sciences': ['psychology', 'sociology', 'economics', 'political science', 'education'],
        'humanities': ['history', 'literature', 'philosophy', 'ethics', 'linguistics'],
        'engineering': ['mechanical engineering', 'electrical engineering', 'civil engineering'],
        'environmental': ['climate change', 'pollution', 'sustainability', 'renewable energy']
    }
    
    category_counts = {}
    for category, terms in categories.items():
        count = sum(filtered_keywords[term] for term in terms if term in filtered_keywords)
        category_counts[category] = count
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    
    # Save category analysis
    with open('merged_keywords/category_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Academic Keyword Categories\n\n")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{category}: {count}\n")
            for term in categories[category]:
                if term in filtered_keywords:
                    f.write(f"  - {term}: {filtered_keywords[term]}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
