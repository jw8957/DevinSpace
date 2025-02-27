import os
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from collections import Counter
import pandas as pd
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

# Fix for NLTK tagger issues
nltk.data.path.append('/home/ubuntu/nltk_data')
os.environ['NLTK_DATA'] = '/home/ubuntu/nltk_data'

# Create output directory
os.makedirs("keywords", exist_ok=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['also', 'many', 'often', 'use', 'used', 'using', 'one', 'two', 'three', 'may', 'first', 'second', 'third'])

# Academic fields and subjects for filtering
academic_fields = [
    # Natural Sciences
    'physics', 'chemistry', 'biology', 'astronomy', 'geology', 'earth science',
    'quantum', 'relativity', 'thermodynamics', 'electromagnetism', 'mechanics',
    'organic chemistry', 'inorganic chemistry', 'biochemistry', 'molecular biology',
    'genetics', 'ecology', 'evolution', 'cell biology', 'microbiology', 'botany',
    'zoology', 'astrophysics', 'cosmology', 'geophysics', 'meteorology', 'oceanography',
    
    # Formal Sciences
    'mathematics', 'statistics', 'logic', 'theoretical computer science',
    'algebra', 'calculus', 'geometry', 'topology', 'number theory', 'analysis',
    'probability', 'discrete mathematics', 'cryptography', 'graph theory',
    'set theory', 'category theory', 'computational complexity', 'automata theory',
    
    # Applied Sciences and Engineering
    'computer science', 'engineering', 'artificial intelligence', 'machine learning',
    'data science', 'robotics', 'cybersecurity', 'software engineering', 'database',
    'networking', 'operating system', 'algorithm', 'data structure', 'programming language',
    'compiler', 'distributed system', 'parallel computing', 'cloud computing',
    'civil engineering', 'mechanical engineering', 'electrical engineering',
    'chemical engineering', 'aerospace engineering', 'biomedical engineering',
    'materials science', 'nanotechnology', 'telecommunications', 'control theory',
    
    # Social Sciences
    'psychology', 'sociology', 'economics', 'political science', 'anthropology',
    'archaeology', 'linguistics', 'cognitive science', 'behavioral economics',
    'macroeconomics', 'microeconomics', 'international relations', 'public policy',
    'urban planning', 'human geography', 'demography', 'criminology',
    
    # Humanities
    'history', 'philosophy', 'literature', 'ethics', 'epistemology', 'metaphysics',
    'aesthetics', 'literary theory', 'historiography', 'cultural studies',
    
    # Medicine and Health Sciences
    'medicine', 'neuroscience', 'pharmacology', 'epidemiology', 'immunology',
    'pathology', 'physiology', 'anatomy', 'cardiology', 'oncology', 'pediatrics',
    'psychiatry', 'surgery', 'public health', 'virology', 'endocrinology',
    
    # Interdisciplinary Fields
    'bioinformatics', 'computational biology', 'cognitive neuroscience',
    'environmental science', 'systems biology', 'network science',
    'information theory', 'operations research', 'game theory', 'complex systems',
    'sustainability', 'renewable energy', 'climate change', 'quantum computing',
    
    # Research Terms
    'theory', 'analysis', 'research', 'study', 'experiment', 'methodology',
    'framework', 'model', 'system', 'network', 'structure', 'function',
    'mechanism', 'process', 'technique', 'algorithm', 'protocol', 'paradigm',
    'hypothesis', 'theorem', 'proof', 'equation', 'formula', 'law', 'principle',
    'method', 'approach', 'implementation', 'evaluation', 'validation',
    'optimization', 'simulation', 'empirical', 'theoretical', 'experimental',
    'quantitative', 'qualitative', 'computational', 'analytical', 'numerical',
    'stochastic', 'deterministic', 'linear', 'nonlinear', 'dynamic', 'static',
    'discrete', 'continuous', 'finite', 'infinite', 'recursive', 'iterative',
    'parallel', 'sequential', 'distributed', 'centralized', 'hierarchical',
    'heterogeneous', 'homogeneous', 'synchronous', 'asynchronous'
]

# Function to extract named entities from text
def extract_named_entities(text):
    entities = []
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            if text is None:
                return []
            text = str(text)
            
        # Skip if text is too short
        if len(text) < 10:
            return []
            
        # Simple named entity extraction based on capitalization patterns
        # This is a fallback method that doesn't rely on NLTK's ne_chunk
        words = word_tokenize(text)
        tagged = pos_tag(words)
        
        # Look for sequences of proper nouns (NNP)
        i = 0
        while i < len(tagged):
            if tagged[i][1] == 'NNP':
                entity = [tagged[i][0]]
                j = i + 1
                while j < len(tagged) and tagged[j][1] == 'NNP':
                    entity.append(tagged[j][0])
                    j += 1
                if len(entity) > 1:  # Only multi-word entities
                    entities.append((' '.join(entity), 'ENTITY'))
                i = j
            else:
                i += 1
                
        # Look for capitalized words that aren't at the beginning of sentences
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for i in range(1, len(words)):  # Skip first word of sentence
                if words[i][0].isupper() and words[i].isalpha() and len(words[i]) > 1:
                    # Check if it's part of a sequence we already found
                    if not any(words[i] in entity for entity, _ in entities):
                        entities.append((words[i], 'ENTITY'))
    except Exception as e:
        print(f"Error in named entity extraction: {e}")
    
    return entities

# Function to extract n-grams from text
def extract_ngrams(text, n=2):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# Function to extract potential academic terms
def extract_academic_terms(text):
    try:
        # Ensure text is a string
        if not isinstance(text, str):
            if text is None:
                return []
            text = str(text)
            
        # Skip if text is too short
        if len(text) < 10:
            return []
            
        # Clean text
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation markers like [1], [2], etc.
        
        # Extract sentences containing academic field keywords
        try:
            sentences = sent_tokenize(text)
            academic_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(field in sentence_lower for field in academic_fields):
                    academic_sentences.append(sentence)
        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            academic_sentences = []
        
        # Extract potential academic terms
        academic_terms = []
        
        # Extract named entities
        try:
            for sentence in academic_sentences:
                entities = extract_named_entities(sentence)
                for entity, entity_type in entities:
                    if entity_type in ['ORGANIZATION', 'PERSON', 'GPE']:
                        continue  # Skip organizations, persons, and geopolitical entities
                    if len(entity.split()) > 1:  # Only multi-word entities
                        academic_terms.append(entity)
        except Exception as e:
            print(f"Error in named entity extraction: {e}")
        
        # Extract noun phrases (potential technical terms)
        try:
            for sentence in academic_sentences:
                words = word_tokenize(sentence)
                tagged = pos_tag(words)
                
                i = 0
                while i < len(tagged):
                    # Look for adjective + noun or noun + noun patterns
                    if i < len(tagged) - 1:
                        if (tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN')) or \
                           (tagged[i][1].startswith('NN') and tagged[i+1][1].startswith('NN')):
                            term = f"{tagged[i][0]} {tagged[i+1][0]}"
                            
                            # Try to extend the noun phrase
                            j = i + 2
                            while j < len(tagged) and tagged[j][1].startswith('NN'):
                                term += f" {tagged[j][0]}"
                                j += 1
                            
                            if term.lower() not in [t.lower() for t in academic_terms]:
                                academic_terms.append(term)
                            
                            i = j
                            continue
                    i += 1
        except Exception as e:
            print(f"Error in noun phrase extraction: {e}")
        
        # Extract bigrams and trigrams
        try:
            bigrams = extract_ngrams(text, 2)
            trigrams = extract_ngrams(text, 3)
            
            # Filter n-grams
            for ngram in bigrams + trigrams:
                ngram_lower = ngram.lower()
                # Check if the n-gram contains an academic field keyword
                if any(field in ngram_lower for field in academic_fields):
                    if ngram not in academic_terms:
                        academic_terms.append(ngram)
        except Exception as e:
            print(f"Error in n-gram extraction: {e}")
        
        return academic_terms
    except Exception as e:
        print(f"Error in academic term extraction: {e}")
        return []

# Function to extract references
def extract_references(text):
    # Look for the References or Bibliography section
    references_section = None
    
    # Common section headers for references
    ref_headers = ['References', 'Bibliography', 'Sources', 'Further reading', 'Notes and references']
    
    # Try different formats of section headers in Wikipedia markup
    for header in ref_headers:
        # Try == Header == format
        match = re.search(rf'== {header} ==(.+?)(?=== [^=]+ ==|$)', text, re.DOTALL)
        if match:
            references_section = match.group(1)
            break
            
        # Try === Header === format
        match = re.search(rf'=== {header} ===(.+?)(?==== [^=]+ ====|=== [^=]+ ===|$)', text, re.DOTALL)
        if match:
            references_section = match.group(1)
            break
    
    if not references_section:
        return []
    
    # Extract titles from references
    ref_titles = []
    
    # Look for titles in quotes or italics
    title_patterns = [
        r'"([^"]+)"',  # Titles in double quotes
        r"'([^']+)'",  # Titles in single quotes
        r"\*([^*]+)\*",  # Titles in italics (markdown style)
        r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]",  # Wiki links [[Title]] or [[Title|Text]]
        r"\{\{cite [^}]*title\s*=\s*([^|}\n]+)",  # Citation templates with title parameter
        r"\{\{citation [^}]*title\s*=\s*([^|}\n]+)"  # Citation templates with title parameter
    ]
    
    for pattern in title_patterns:
        titles = re.findall(pattern, references_section)
        ref_titles.extend(titles)
    
    # Extract DOI, arXiv, and other identifiers which often contain keywords in their titles
    doi_matches = re.findall(r'doi:([^\s]+)', references_section)
    arxiv_matches = re.findall(r'arXiv:([^\s]+)', references_section)
    
    # Extract journal names which are often good sources of academic keywords
    journal_patterns = [
        r"journal\s*=\s*([^|}\n]+)",  # Journal parameter in citation templates
        r"publisher\s*=\s*([^|}\n]+)",  # Publisher parameter in citation templates
        r"conference\s*=\s*([^|}\n]+)"  # Conference parameter in citation templates
    ]
    
    journal_names = []
    for pattern in journal_patterns:
        matches = re.findall(pattern, references_section)
        journal_names.extend(matches)
    
    # Extract potential academic terms from reference titles and journal names
    ref_terms = []
    
    # Process titles
    for title in ref_titles:
        if len(title) > 5:  # Ignore very short titles
            # Direct extraction of key phrases from titles
            words = title.split()
            if 2 <= len(words) <= 5:  # Titles with 2-5 words might be good keywords themselves
                ref_terms.append(title)
            
            # Extract academic terms using our function
            terms = extract_academic_terms(title)
            ref_terms.extend(terms)
    
    # Process journal names
    for journal in journal_names:
        if len(journal) > 3:  # Ignore very short journal names
            # Clean up journal names
            journal = re.sub(r'[\[\]{}"]', '', journal).strip()
            if journal and not any(term.lower() == journal.lower() for term in ref_terms):
                ref_terms.append(journal)
    
    return ref_terms

# Function to process a single Wikipedia article
def process_article(article):
    title = article.get('title', '')
    text = article.get('text', '')
    url = article.get('url', '')
    
    # Skip disambiguation pages, lists, and non-content pages
    if any(x in title for x in ['disambiguation', 'List of', 'Index of', 'Outline of', 'Category:', 'Template:', 'File:', 'Wikipedia:', 'Portal:', 'Help:']):
        return []
    
    # Extract academic terms from the main text
    academic_terms = extract_academic_terms(text)
    
    # Extract terms from references
    reference_terms = extract_references(text)
    
    # Add the article title itself if it's a potential academic term
    title_words = title.split()
    if 2 <= len(title_words) <= 5 and not any(char in title for char in [':', '/', '\\']):
        # Check if title contains any academic field keywords
        title_lower = title.lower()
        if any(field in title_lower for field in academic_fields):
            academic_terms.append(title)
    
    # Combine and deduplicate terms
    all_terms = academic_terms + reference_terms
    unique_terms = list(set(all_terms))
    
    # Filter out terms that are too short or too long
    filtered_terms = [term for term in unique_terms if 2 <= len(term.split()) <= 5]
    
    # Additional filtering for quality
    high_quality_terms = []
    for term in filtered_terms:
        term_lower = term.lower()
        
        # Skip terms with numbers unless they appear to be versions or years
        if any(char.isdigit() for char in term):
            # Allow version numbers (e.g., "Web 2.0", "Python 3.8")
            if not re.search(r'\b\d+\.\d+\b', term) and not re.search(r'\b(19|20)\d{2}\b', term):
                continue
        
        # Skip terms with special characters
        if re.search(r'[^\w\s\.-]', term):
            continue
            
        # Skip terms that are likely not academic
        if any(word in term_lower for word in ['best', 'top', 'worst', 'how to', 'guide', 'tutorial']):
            continue
            
        # Skip terms that are too generic
        if term_lower in ['the system', 'this method', 'the process', 'the theory']:
            continue
            
        high_quality_terms.append(term)
    
    return high_quality_terms

# Function to process a chunk of Wikipedia articles
def process_chunk(chunk_file, output_file):
    print(f"Processing {chunk_file}...")
    
    # Load the chunk (handling JSONL format where each line is a separate JSON object)
    articles = []
    with open(chunk_file, 'r') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
                continue
    
    print(f"Loaded {len(articles)} articles from {chunk_file}")
    all_keywords = []
    
    # Process each article
    for i, article in enumerate(articles):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(articles)} articles...")
        
        keywords = process_article(article)
        all_keywords.extend(keywords)
    
    # Count keyword frequencies
    keyword_counts = Counter(all_keywords)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(keyword_counts.items(), columns=['keyword', 'frequency'])
    df = df.sort_values('frequency', ascending=False)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Saved {len(df)} keywords to {output_file}")
    return df

# Main function to process all chunks
def main():
    print("Starting keyword extraction...")
    
    # Process only the sample file first to test the script
    if os.path.exists("english_wikipedia/sample.json"):
        print("Processing sample file...")
        try:
            sample_df = process_chunk("english_wikipedia/sample.json", "keywords/sample_keywords.csv")
            print(f"Top 20 keywords from sample:\n{sample_df.head(20)}")
            print("Sample processing successful!")
        except Exception as e:
            print(f"Error processing sample file: {e}")
            return
    else:
        print("Sample file not found. Make sure the download has completed.")
        return
    
    # Ask if we should continue with processing all chunks
    print("\nSample processing completed successfully.")
    print("Do you want to process all chunks? This may take a long time.")
    print("For now, we'll process just a few chunks to get a representative set of keywords.")
    
    # Process a subset of chunk files (first 5 chunks)
    chunk_files = [f for f in os.listdir("english_wikipedia") if f.startswith("chunk_") and f.endswith(".json")]
    
    if not chunk_files:
        print("No chunk files found. Make sure the download has completed.")
        return
    
    # Sort chunk files to process them in order
    chunk_files.sort(key=lambda x: int(x.split('_')[1]))
    
    # Process only the first 5 chunks for now
    subset_chunks = chunk_files[:5]
    print(f"Processing {len(subset_chunks)} chunks out of {len(chunk_files)} total chunks...")
    
    all_dfs = []
    
    for i, chunk_file in enumerate(subset_chunks):
        print(f"Processing chunk {i+1}/{len(subset_chunks)}...")
        chunk_path = os.path.join("english_wikipedia", chunk_file)
        output_path = os.path.join("keywords", f"keywords_{i+1}.csv")
        
        try:
            df = process_chunk(chunk_path, output_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing chunk {chunk_file}: {e}")
            continue
    
    # Combine all keyword dataframes
    if all_dfs:
        print("Combining all keywords...")
        combined_df = pd.concat(all_dfs)
        
        # Aggregate frequencies
        combined_df = combined_df.groupby('keyword')['frequency'].sum().reset_index()
        combined_df = combined_df.sort_values('frequency', ascending=False)
        
        # Save combined keywords
        combined_df.to_csv("keywords/all_keywords.csv", index=False)
        
        print(f"Total unique keywords: {len(combined_df)}")
        print(f"Top 50 keywords:\n{combined_df.head(50)}")
        
        # Save top 1000 keywords to a separate file for easy review
        top_keywords = combined_df.head(1000)
        top_keywords.to_csv("keywords/top_1000_keywords.csv", index=False)
        print(f"Saved top 1000 keywords to keywords/top_1000_keywords.csv")
    
    print("Keyword extraction complete!")

if __name__ == "__main__":
    main()
