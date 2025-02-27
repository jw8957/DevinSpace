import os
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
import string

# Download necessary NLTK resources (only the ones we know work)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create output directory
os.makedirs("keywords", exist_ok=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['also', 'many', 'often', 'use', 'used', 'using', 'one', 'two', 'three', 'may', 'first', 'second', 'third'])

# Academic fields and subjects for filtering - more specific terms
academic_fields = [
    # Computer Science & AI
    'machine learning', 'deep learning', 'neural network', 'artificial intelligence', 'natural language processing',
    'computer vision', 'data mining', 'big data', 'cloud computing', 'distributed computing',
    'parallel computing', 'quantum computing', 'blockchain', 'cryptography', 'cybersecurity',
    'information retrieval', 'knowledge representation', 'semantic web', 'computer graphics',
    'human-computer interaction', 'software engineering', 'algorithm', 'data structure',
    'operating system', 'database', 'network protocol', 'compiler', 'programming language',
    
    # Physics
    'quantum mechanics', 'relativity', 'string theory', 'particle physics', 'nuclear physics',
    'condensed matter', 'astrophysics', 'cosmology', 'thermodynamics', 'electromagnetism',
    'optics', 'quantum field theory', 'statistical mechanics', 'fluid dynamics', 'plasma physics',
    'atomic physics', 'molecular physics', 'quantum gravity', 'quantum electrodynamics',
    'superconductivity', 'quantum entanglement', 'higgs boson', 'dark matter', 'dark energy',
    
    # Biology & Medicine
    'molecular biology', 'cell biology', 'genetics', 'genomics', 'proteomics',
    'bioinformatics', 'neuroscience', 'immunology', 'microbiology', 'virology',
    'biochemistry', 'biotechnology', 'ecology', 'evolution', 'developmental biology',
    'systems biology', 'synthetic biology', 'cancer research', 'stem cell', 'gene therapy',
    'personalized medicine', 'epidemiology', 'public health', 'pharmacology', 'toxicology',
    
    # Chemistry
    'organic chemistry', 'inorganic chemistry', 'physical chemistry', 'analytical chemistry',
    'polymer chemistry', 'medicinal chemistry', 'computational chemistry', 'electrochemistry',
    'photochemistry', 'nuclear chemistry', 'surface chemistry', 'catalysis', 'chemical kinetics',
    'chemical thermodynamics', 'spectroscopy', 'chromatography', 'mass spectrometry',
    
    # Mathematics
    'algebra', 'geometry', 'calculus', 'topology', 'number theory', 'analysis',
    'differential equation', 'probability theory', 'statistics', 'discrete mathematics',
    'graph theory', 'set theory', 'category theory', 'computational complexity',
    'automata theory', 'game theory', 'optimization', 'numerical analysis',
    'mathematical modeling', 'dynamical systems', 'chaos theory', 'fractal',
    
    # Engineering
    'civil engineering', 'mechanical engineering', 'electrical engineering', 'chemical engineering',
    'aerospace engineering', 'biomedical engineering', 'environmental engineering',
    'materials science', 'nanotechnology', 'robotics', 'control theory', 'signal processing',
    'image processing', 'computer-aided design', 'finite element analysis', 'computational fluid dynamics',
    
    # Social Sciences
    'psychology', 'cognitive psychology', 'behavioral economics', 'neurolinguistics',
    'sociolinguistics', 'anthropology', 'archaeology', 'sociology', 'political science',
    'international relations', 'public policy', 'urban planning', 'human geography',
    'demography', 'criminology', 'economic theory', 'macroeconomics', 'microeconomics',
    
    # Humanities
    'philosophy of science', 'philosophy of mind', 'epistemology', 'metaphysics',
    'ethics', 'aesthetics', 'literary theory', 'historiography', 'cultural studies',
    'digital humanities', 'comparative literature', 'linguistics', 'semiotics',
    
    # Interdisciplinary Fields
    'cognitive science', 'computational linguistics', 'computational biology',
    'environmental science', 'network science', 'information theory',
    'operations research', 'complex systems', 'sustainability', 'renewable energy',
    'climate change', 'data science', 'quantum information', 'systems theory',
    
    # Research Methodologies
    'experimental design', 'clinical trial', 'meta-analysis', 'systematic review',
    'qualitative research', 'quantitative research', 'mixed methods', 'longitudinal study',
    'cross-sectional study', 'randomized controlled trial', 'case study', 'ethnography',
    'grounded theory', 'action research', 'participatory research', 'survey methodology',
    'statistical analysis', 'machine learning algorithm', 'deep learning model',
    
    # Specific Technologies and Techniques
    'mapreduce', 'hadoop', 'spark', 'tensorflow', 'pytorch', 'convolutional neural network',
    'recurrent neural network', 'transformer model', 'bert model', 'gpt model',
    'reinforcement learning', 'unsupervised learning', 'supervised learning',
    'transfer learning', 'federated learning', 'generative adversarial network',
    'natural language understanding', 'computer vision system', 'autonomous vehicle',
    'internet of things', 'augmented reality', 'virtual reality', 'mixed reality',
    '3d printing', 'crispr', 'gene editing', 'renewable energy', 'smart grid',
    'blockchain technology', 'cryptocurrency', 'quantum cryptography'
]

# Specific academic phrases that are likely to be good keywords
specific_academic_phrases = [
    'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
    'natural language processing', 'computer vision', 'quantum mechanics',
    'general relativity', 'special relativity', 'string theory', 'quantum field theory',
    'molecular biology', 'genetic algorithm', 'climate change', 'renewable energy',
    'sustainable development', 'public health', 'cognitive science', 'game theory',
    'information theory', 'data mining', 'big data', 'cloud computing',
    'internet of things', 'blockchain technology', 'virtual reality', 'augmented reality',
    'computational linguistics', 'digital humanities', 'social network analysis',
    'human-computer interaction', 'computer-aided design', 'finite element analysis',
    'computational fluid dynamics', 'systems biology', 'synthetic biology',
    'personalized medicine', 'evidence-based medicine', 'randomized controlled trial',
    'meta-analysis', 'systematic review', 'qualitative research', 'quantitative research',
    'mixed methods research', 'grounded theory', 'case study research', 'ethnography',
    'discourse analysis', 'content analysis', 'narrative analysis', 'phenomenology',
    'hermeneutics', 'critical theory', 'postcolonial theory', 'feminist theory',
    'queer theory', 'cultural studies', 'media studies', 'science and technology studies',
    'philosophy of science', 'philosophy of mind', 'philosophy of language',
    'cognitive psychology', 'developmental psychology', 'social psychology',
    'clinical psychology', 'neuropsychology', 'behavioral economics',
    'experimental economics', 'international relations', 'comparative politics',
    'public administration', 'public policy analysis', 'urban planning',
    'environmental policy', 'educational research', 'learning theory',
    'instructional design', 'educational technology', 'special education',
    'higher education', 'curriculum development', 'assessment and evaluation',
    'organizational behavior', 'strategic management', 'operations management',
    'supply chain management', 'marketing research', 'consumer behavior',
    'financial economics', 'corporate finance', 'investment analysis',
    'risk management', 'actuarial science', 'econometrics', 'time series analysis',
    'panel data analysis', 'cross-sectional analysis', 'longitudinal analysis',
    'spatial analysis', 'geographic information system', 'remote sensing',
    'environmental monitoring', 'conservation biology', 'restoration ecology',
    'landscape ecology', 'ecosystem services', 'biodiversity conservation',
    'climate modeling', 'atmospheric science', 'oceanography', 'hydrology',
    'geomorphology', 'paleoclimatology', 'quaternary science', 'geochronology',
    'isotope analysis', 'radiometric dating', 'archaeological science',
    'bioarchaeology', 'zooarchaeology', 'paleoanthropology', 'historical linguistics',
    'comparative linguistics', 'corpus linguistics', 'computational linguistics',
    'applied linguistics', 'second language acquisition', 'translation studies',
    'interpreting studies', 'discourse analysis', 'conversation analysis',
    'critical discourse analysis', 'multimodal analysis', 'visual analysis',
    'sound studies', 'performance studies', 'digital media', 'new media',
    'social media analysis', 'network analysis', 'bibliometric analysis',
    'scientometric analysis', 'altmetric analysis', 'research evaluation',
    'science policy', 'innovation studies', 'technology assessment',
    'responsible research and innovation', 'research ethics', 'bioethics',
    'neuroethics', 'environmental ethics', 'business ethics', 'professional ethics',
    'engineering ethics', 'computer ethics', 'information ethics', 'data ethics',
    'privacy and surveillance', 'intellectual property', 'copyright law',
    'patent law', 'trademark law', 'media law', 'internet law', 'cybersecurity law',
    'health law', 'medical law', 'biolaw', 'environmental law', 'energy law',
    'climate law', 'international environmental law', 'human rights law',
    'humanitarian law', 'migration law', 'refugee law', 'labor law',
    'employment law', 'social security law', 'tax law', 'financial regulation',
    'banking law', 'securities regulation', 'antitrust law', 'competition law',
    'consumer protection law', 'food safety regulation', 'drug regulation',
    'medical device regulation', 'healthcare regulation', 'public health law',
    'global health governance', 'international health regulations'
]

# Function to extract n-grams from text
def extract_ngrams(text, n=2):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# Function to extract potential academic terms using improved pattern matching
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
        
        # Direct matching of specific academic phrases
        for phrase in specific_academic_phrases:
            if phrase in text.lower():
                academic_terms.append(phrase)
        
        # Extract phrases containing academic field keywords
        try:
            for sentence in academic_sentences:
                words = word_tokenize(sentence)
                
                for i in range(len(words) - 1):
                    for j in range(i + 1, min(i + 6, len(words))):  # Look for phrases up to 5 words
                        phrase = ' '.join(words[i:j+1])
                        phrase_lower = phrase.lower()
                        
                        # Check if the phrase contains an academic field keyword
                        if any(field in phrase_lower for field in academic_fields):
                            # Clean the phrase
                            phrase = re.sub(r'[^\w\s\.-]', '', phrase)
                            if phrase and phrase not in academic_terms:
                                academic_terms.append(phrase)
        except Exception as e:
            print(f"Error in academic phrase extraction: {e}")
        
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

# Function to clean and normalize keywords
def clean_keywords(keywords):
    cleaned_keywords = []
    
    for keyword in keywords:
        # Convert to lowercase
        keyword = keyword.lower()
        
        # Remove punctuation at the beginning and end
        keyword = keyword.strip(string.punctuation + ' ')
        
        # Skip if keyword is too short or contains unwanted characters
        if len(keyword) < 3 or not keyword or re.search(r'[^\w\s\.-]', keyword):
            continue
            
        # Skip if keyword is just a single common word
        if len(keyword.split()) == 1 and keyword in stop_words:
            continue
            
        # Skip if keyword is a fragment or incomplete phrase
        if keyword.startswith(('the ', 'a ', 'an ', 'of ', 'in ', 'on ', 'at ', 'to ', 'for ', 'with ')):
            continue
            
        # Skip if keyword ends with common prepositions or articles
        if keyword.endswith((' of', ' in', ' on', ' at', ' to', ' for', ' with', ' the', ' a', ' an')):
            continue
            
        # Skip if keyword contains only common words
        words = keyword.split()
        if all(word in stop_words for word in words):
            continue
            
        # Skip very generic academic terms
        generic_terms = ['history of', 'study of', 'system of', 'theory of', 'analysis of', 
                         'process of', 'method of', 'function of', 'structure of', 'principle of']
        if keyword in generic_terms:
            continue
            
        # Skip generic standalone terms and phrases identified as not useful
        generic_standalone_terms = [
            'analysis', 'statistics', 'revolution', 'evolution', 'psychology', 'ethics', 
            'geometry', 'database', 'linguistics', 'aesthetics'
        ]
        if keyword in generic_standalone_terms:
            continue
            
        # Skip generic phrases with conjunctions that are too vague
        generic_phrases = [
            'analysis and', 'and analysis', 'statistics are', 'official statistics',
            'according statistics', 'according to statistics', 'statistics from',
            'revolution and', 'and revolution', 'evolution and', 'and evolution',
            'psychology and', 'and psychology', 'ethics and', 'and ethics'
        ]
        if any(keyword == phrase or keyword.startswith(phrase + ' ') or keyword.endswith(' ' + phrase) for phrase in generic_phrases):
            continue
            
        # Skip if keyword is just a single word (unless it's a specific technical term)
        if len(keyword.split()) == 1:
            specific_technical_terms = [
                'mapreduce', 'hadoop', 'spark', 'tensorflow', 'pytorch', 'crispr',
                'blockchain', 'cryptocurrency', 'nanotechnology', 'robotics',
                'genomics', 'proteomics', 'bioinformatics', 'neuroscience',
                'immunology', 'microbiology', 'virology', 'biochemistry',
                'biotechnology', 'ecology', 'epidemiology', 'pharmacology',
                'toxicology', 'spectroscopy', 'chromatography', 'catalysis',
                'algebra', 'calculus', 'topology', 'cryptography', 'cybersecurity',
                'cosmology', 'astrophysics', 'thermodynamics', 'electromagnetism',
                'optics', 'superconductivity', 'entanglement', 'photonics',
                'metamaterials', 'nanophotonics', 'plasmonics', 'spintronics',
                'graphene', 'fullerene', 'nanotube', 'quantum', 'relativity',
                'higgs', 'fermion', 'boson', 'quark', 'lepton', 'hadron',
                'photon', 'gluon', 'neutrino', 'graviton', 'meson', 'baryon'
            ]
            if keyword.lower() not in [t.lower() for t in specific_technical_terms]:
                continue
                
        # Add to cleaned keywords if not already present
        if keyword not in cleaned_keywords:
            cleaned_keywords.append(keyword)
    
    return cleaned_keywords

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
    
    # Clean and normalize keywords
    cleaned_terms = clean_keywords(all_terms)
    
    # Additional filtering for quality
    high_quality_terms = []
    for term in cleaned_terms:
        # Skip terms with numbers unless they appear to be versions or years
        if any(char.isdigit() for char in term):
            # Allow version numbers (e.g., "Web 2.0", "Python 3.8")
            if not re.search(r'\b\d+\.\d+\b', term) and not re.search(r'\b(19|20)\d{2}\b', term):
                continue
            
        # Skip terms that are likely not academic
        if any(word in term for word in ['best', 'top', 'worst', 'how to', 'guide', 'tutorial']):
            continue
            
        # Skip terms that are too generic
        generic_standalone_terms = [
            'the system', 'this method', 'the process', 'the theory', 
            'analysis', 'statistics', 'revolution', 'evolution', 'psychology', 
            'ethics', 'geometry', 'database', 'linguistics', 'aesthetics',
            'theory', 'method', 'process', 'system', 'structure', 'function',
            'principle', 'concept', 'model', 'framework', 'approach', 'technique',
            'mechanism', 'paradigm', 'perspective', 'dimension', 'aspect', 'factor',
            'element', 'component', 'variable', 'parameter', 'metric', 'measure',
            'indicator', 'index', 'ratio', 'rate', 'frequency', 'distribution',
            'correlation', 'regression', 'classification', 'clustering', 'prediction',
            'estimation', 'inference', 'hypothesis', 'experiment', 'observation',
            'measurement', 'calculation', 'computation', 'simulation', 'modeling',
            'optimization', 'evaluation', 'assessment', 'validation', 'verification'
        ]
        if term in generic_standalone_terms:
            continue
            
        # Skip generic phrases with conjunctions that are too vague
        generic_phrases = [
            'analysis and', 'and analysis', 'statistics are', 'official statistics',
            'according statistics', 'according to statistics', 'statistics from',
            'revolution and', 'and revolution', 'evolution and', 'and evolution',
            'psychology and', 'and psychology', 'ethics and', 'and ethics',
            'analysis is', 'is analysis', 'statistics is', 'is statistics',
            'theory of', 'method of', 'process of', 'system of', 'structure of',
            'function of', 'principle of', 'concept of', 'model of', 'framework of',
            'approach to', 'technique for', 'mechanism of', 'paradigm of', 'perspective on',
            'dimension of', 'aspect of', 'factor in', 'element of', 'component of',
            'variable in', 'parameter of', 'metric for', 'measure of', 'indicator of',
            'index of', 'ratio of', 'rate of', 'frequency of', 'distribution of',
            'correlation between', 'regression of', 'classification of', 'clustering of',
            'prediction of', 'estimation of', 'inference about', 'hypothesis on',
            'experiment on', 'observation of', 'measurement of', 'calculation of',
            'computation of', 'simulation of', 'modeling of', 'optimization of',
            'evaluation of', 'assessment of', 'validation of', 'verification of'
        ]
        if any(term == phrase or term.startswith(phrase + ' ') or term.endswith(' ' + phrase) for phrase in generic_phrases):
            continue
            
        # Prioritize terms that match our specific academic phrases
        if term in specific_academic_phrases:
            high_quality_terms.append(term)
            continue
            
        # Check if term contains any of our academic field keywords
        if any(field in term for field in academic_fields):
            high_quality_terms.append(term)
            continue
            
        # Include terms that are likely to be academic based on word count
        # For single words, only include if they are very specific technical terms
        if len(term.split()) == 1:
            specific_technical_terms = [
                'mapreduce', 'hadoop', 'spark', 'tensorflow', 'pytorch', 'crispr',
                'blockchain', 'cryptocurrency', 'nanotechnology', 'robotics',
                'genomics', 'proteomics', 'bioinformatics', 'neuroscience',
                'immunology', 'microbiology', 'virology', 'biochemistry',
                'biotechnology', 'ecology', 'epidemiology', 'pharmacology',
                'toxicology', 'spectroscopy', 'chromatography', 'catalysis',
                'algebra', 'calculus', 'topology', 'cryptography', 'cybersecurity',
                'blockchain', 'cryptocurrency', 'nanotechnology', 'robotics',
                'genomics', 'proteomics', 'bioinformatics', 'neuroscience',
                'immunology', 'microbiology', 'virology', 'biochemistry',
                'biotechnology', 'ecology', 'epidemiology', 'pharmacology',
                'toxicology', 'spectroscopy', 'chromatography', 'catalysis',
                'algebra', 'calculus', 'topology', 'cryptography', 'cybersecurity',
                'cosmology', 'astrophysics', 'thermodynamics', 'electromagnetism',
                'optics', 'superconductivity', 'entanglement', 'photonics',
                'metamaterials', 'nanophotonics', 'plasmonics', 'spintronics',
                'graphene', 'fullerene', 'nanotube', 'quantum', 'relativity',
                'higgs', 'fermion', 'boson', 'quark', 'lepton', 'hadron',
                'photon', 'gluon', 'neutrino', 'graviton', 'meson', 'baryon'
            ]
            if term.lower() in [t.lower() for t in specific_technical_terms]:
                high_quality_terms.append(term)
        # For multi-word terms, include if they are 2-5 words
        elif 2 <= len(term.split()) <= 5:
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
    print("Starting improved keyword extraction...")
    
    # Process only the sample file first to test the script
    if os.path.exists("english_wikipedia/sample.json"):
        print("Processing sample file...")
        try:
            # Process only the first 100 articles from the sample for quick testing
            print("Loading sample file...")
            articles = []
            with open("english_wikipedia/sample.json", 'r') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Only process first 100 articles
                        break
                    try:
                        article = json.loads(line.strip())
                        articles.append(article)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {i}: {e}")
                        continue
            
            print(f"Processing {len(articles)} articles from sample...")
            all_keywords = []
            
            # Process each article with verbose output
            for i, article in enumerate(articles):
                print(f"Processing article {i+1}/{len(articles)}: {article.get('title', 'Unknown')}")
                
                keywords = process_article(article)
                if keywords:
                    print(f"  Found {len(keywords)} keywords: {', '.join(keywords[:5])}...")
                else:
                    print("  No keywords found")
                
                all_keywords.extend(keywords)
            
            # Count keyword frequencies
            keyword_counts = Counter(all_keywords)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(keyword_counts.items(), columns=['keyword', 'frequency'])
            df = df.sort_values('frequency', ascending=False)
            
            # Save to CSV
            df.to_csv("keywords/improved_sample_keywords.csv", index=False)
            
            print(f"Saved {len(df)} keywords to keywords/improved_sample_keywords.csv")
            print(f"Top 20 keywords from sample:\n{df.head(20)}")
            print("Sample processing successful!")
            
            # Save top 1000 keywords to a separate file for easy review
            top_keywords = df.head(1000)
            top_keywords.to_csv("keywords/improved_top_1000_keywords.csv", index=False)
            print(f"Saved top keywords to keywords/improved_top_1000_keywords.csv")
            
        except Exception as e:
            print(f"Error processing sample file: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("Sample file not found. Make sure the download has completed.")
        return
    
    print("Keyword extraction complete!")
    
    # Note: We're only processing the sample file for now to verify the script works correctly
    # Once verified, we can uncomment the code below to process more chunks
    
    
    # Process a subset of chunk files (first 5 chunks)
    # Uncomment the code below to process more chunks
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
        output_path = os.path.join("keywords", f"improved_keywords_{i+1}.csv")
        
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
        combined_df.to_csv("keywords/improved_all_keywords.csv", index=False)
        
        print(f"Total unique keywords: {len(combined_df)}")
        print(f"Top 50 keywords:\n{combined_df.head(50)}")
    

if __name__ == "__main__":
    main()
