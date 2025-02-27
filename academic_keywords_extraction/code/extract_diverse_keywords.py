import os
import json
import re
import csv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter, defaultdict
import string

# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'words']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

# Define academic fields for filtering
ACADEMIC_FIELDS = [
    # Physics
    'quantum mechanics', 'relativity', 'particle physics', 'astrophysics', 'thermodynamics',
    'nuclear physics', 'condensed matter', 'optics', 'electromagnetism', 'fluid dynamics',
    
    # Biology
    'molecular biology', 'genetics', 'ecology', 'evolution', 'microbiology',
    'biochemistry', 'cell biology', 'neuroscience', 'immunology', 'physiology',
    
    # Medicine
    'epidemiology', 'pharmacology', 'pathology', 'oncology', 'cardiology',
    'neurology', 'immunotherapy', 'radiology', 'surgery', 'pediatrics',
    
    # Chemistry
    'organic chemistry', 'inorganic chemistry', 'analytical chemistry', 'physical chemistry',
    'biochemistry', 'polymer chemistry', 'medicinal chemistry', 'electrochemistry',
    
    # Computer Science
    'artificial intelligence', 'machine learning', 'computer vision', 'natural language processing',
    'algorithms', 'data structures', 'computer networks', 'cybersecurity', 'database systems',
    
    # Mathematics
    'algebra', 'geometry', 'calculus', 'topology', 'number theory',
    'statistics', 'probability', 'discrete mathematics', 'mathematical analysis',
    
    # Social Sciences
    'psychology', 'sociology', 'anthropology', 'economics', 'political science',
    'linguistics', 'archaeology', 'geography', 'demography', 'education',
    
    # Humanities
    'philosophy', 'history', 'literature', 'art history', 'religious studies',
    'ethics', 'cultural studies', 'linguistics', 'archaeology', 'musicology',
    
    # Engineering
    'mechanical engineering', 'electrical engineering', 'civil engineering', 'chemical engineering',
    'aerospace engineering', 'biomedical engineering', 'environmental engineering',
    
    # Earth Sciences
    'geology', 'meteorology', 'oceanography', 'atmospheric science', 'seismology',
    'hydrology', 'climatology', 'geophysics', 'paleontology', 'volcanology',
    
    # Environmental Sciences
    'ecology', 'conservation biology', 'environmental chemistry', 'climate change',
    'sustainability', 'renewable energy', 'pollution', 'biodiversity',
    
    # Interdisciplinary Fields
    'bioinformatics', 'nanotechnology', 'cognitive science', 'systems biology',
    'computational linguistics', 'bioethics', 'data science', 'network science'
]

# Generic terms to filter out
GENERIC_TERMS = [
    'analysis', 'statistics', 'research', 'study', 'method', 'approach', 'theory',
    'model', 'framework', 'concept', 'process', 'system', 'technique', 'structure',
    'function', 'development', 'application', 'implementation', 'evaluation',
    'assessment', 'investigation', 'examination', 'exploration', 'review',
    'analysis and', 'statistics are', 'research on', 'study of', 'method for',
    'approach to', 'theory of', 'model of', 'framework for', 'concept of',
    'process of', 'system for', 'technique for', 'structure of', 'function of',
    'development of', 'application of', 'implementation of', 'evaluation of',
    'assessment of', 'investigation of', 'examination of', 'exploration of', 'review of'
]

# Extract n-grams from text
def extract_ngrams(text, n_range=(2, 5)):
    if not text or not isinstance(text, str):
        return []
    
    # Clean text
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    # Extract n-grams
    all_ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        n_grams = list(ngrams(tokens, n))
        all_ngrams.extend([' '.join(gram) for gram in n_grams])
    
    # Also include single words that match academic fields
    for token in tokens:
        if any(token in field.split() for field in ACADEMIC_FIELDS):
            all_ngrams.append(token)
    
    return all_ngrams

# Filter keywords to keep only academic terms
def filter_academic_keywords(keywords_counter):
    filtered_keywords = Counter()
    
    for keyword, count in keywords_counter.items():
        # Skip generic terms
        if keyword.lower() in GENERIC_TERMS:
            continue
            
        # Skip very short keywords (except those that are part of academic fields)
        if len(keyword.split()) == 1 and not any(keyword in field.split() for field in ACADEMIC_FIELDS):
            continue
        
        # Skip keywords with generic patterns
        if re.match(r'^(the|a|an|and|or|of|in|on|at|by|for|with|to)\b', keyword):
            continue
            
        # Keep keywords that contain academic field terms
        if any(field in keyword.lower() for field in ACADEMIC_FIELDS):
            filtered_keywords[keyword] = count
            continue
            
        # Keep multi-word keywords that look academic
        if len(keyword.split()) >= 2:
            # Check if it contains academic-sounding words
            academic_words = ['theory', 'algorithm', 'method', 'model', 'system', 'analysis',
                             'function', 'structure', 'process', 'mechanism', 'technique']
            if any(word in keyword.lower().split() for word in academic_words):
                filtered_keywords[keyword] = count
    
    return filtered_keywords

# Process PubMed/ArXiv abstracts dataset
def process_pubmed_arxiv_abstracts(file_path):
    print(f"Processing {file_path}...")
    keywords_counter = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Extract text from abstract and title
            abstract = item.get('abstr', '')
            title = item.get('title', '')
            field = item.get('field', '')
            
            # Extract n-grams from abstract and title
            abstract_ngrams = extract_ngrams(abstract)
            title_ngrams = extract_ngrams(title)
            
            # Count keywords
            keywords_counter.update(abstract_ngrams)
            keywords_counter.update(title_ngrams)
            
            # Add field as a keyword if it's not generic
            if field and field.lower() not in GENERIC_TERMS:
                keywords_counter[field.lower()] += 1
        
        # Filter keywords
        filtered_keywords = filter_academic_keywords(keywords_counter)
        return filtered_keywords
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return Counter()

# Process processed PubMed papers dataset
def process_pubmed_papers(file_path):
    print(f"Processing {file_path}...")
    keywords_counter = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Extract text from abstract and summary
            abstract = item.get('abstract', '')
            summary = item.get('summary', '')
            
            # Extract n-grams from abstract and summary
            abstract_ngrams = extract_ngrams(abstract)
            summary_ngrams = extract_ngrams(summary)
            
            # Count keywords
            keywords_counter.update(abstract_ngrams)
            keywords_counter.update(summary_ngrams)
        
        # Filter keywords
        filtered_keywords = filter_academic_keywords(keywords_counter)
        return filtered_keywords
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return Counter()

# Process scientific papers archive dataset
def process_scientific_papers(file_path):
    print(f"Processing {file_path}...")
    keywords_counter = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Extract text from input and output
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # Extract abstract from input (if available)
            abstract_match = re.search(r'abstract for a scientific paper:(.*?)(?:\*|And you have)', input_text, re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                abstract_ngrams = extract_ngrams(abstract)
                keywords_counter.update(abstract_ngrams)
            
            # Extract n-grams from output
            output_ngrams = extract_ngrams(output_text)
            keywords_counter.update(output_ngrams)
        
        # Filter keywords
        filtered_keywords = filter_academic_keywords(keywords_counter)
        return filtered_keywords
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return Counter()

# Save keywords to CSV file
def save_keywords_to_csv(keywords_counter, output_file):
    print(f"Saving {len(keywords_counter)} keywords to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keyword', 'Frequency'])
        
        for keyword, count in keywords_counter.most_common():
            writer.writerow([keyword, count])

# Main function
def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # Create output directory
    os.makedirs('keywords', exist_ok=True)
    
    # Process each dataset
    pubmed_arxiv_keywords = process_pubmed_arxiv_abstracts('pubmed_arxiv_abstracts/sample.json')
    pubmed_papers_keywords = process_pubmed_papers('processed_pubmed/sample.json')
    scientific_papers_keywords = process_scientific_papers('scientific_papers_archive/sample.json')
    
    # Save individual keyword sets
    save_keywords_to_csv(pubmed_arxiv_keywords, 'keywords/pubmed_arxiv_keywords.csv')
    save_keywords_to_csv(pubmed_papers_keywords, 'keywords/pubmed_papers_keywords.csv')
    save_keywords_to_csv(scientific_papers_keywords, 'keywords/scientific_papers_keywords.csv')
    
    # Combine all keywords
    combined_keywords = Counter()
    combined_keywords.update(pubmed_arxiv_keywords)
    combined_keywords.update(pubmed_papers_keywords)
    combined_keywords.update(scientific_papers_keywords)
    
    # Save combined keywords
    save_keywords_to_csv(combined_keywords, 'keywords/diverse_academic_keywords.csv')
    
    # Save top keywords
    top_1000_keywords = Counter(dict(combined_keywords.most_common(1000)))
    top_100_keywords = Counter(dict(combined_keywords.most_common(100)))
    top_50_keywords = Counter(dict(combined_keywords.most_common(50)))
    
    save_keywords_to_csv(top_1000_keywords, 'keywords/diverse_top_1000_keywords.csv')
    save_keywords_to_csv(top_100_keywords, 'keywords/diverse_top_100_keywords.csv')
    save_keywords_to_csv(top_50_keywords, 'keywords/diverse_top_50_keywords.csv')
    
    # Print summary
    print("\nKeyword extraction complete!")
    print(f"Total unique keywords: {len(combined_keywords)}")
    print(f"PubMed/ArXiv abstracts keywords: {len(pubmed_arxiv_keywords)}")
    print(f"Processed PubMed papers keywords: {len(pubmed_papers_keywords)}")
    print(f"Scientific papers archive keywords: {len(scientific_papers_keywords)}")
    
    # Print top 20 keywords
    print("\nTop 20 keywords:")
    for keyword, count in combined_keywords.most_common(20):
        print(f"  {keyword}: {count}")

if __name__ == "__main__":
    main()
