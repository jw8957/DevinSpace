# Academic Keywords Extraction

This project extracts academic keywords from various sources for Google Scholar search, with the goal of finding around 1 million relevant papers.

## Overview

The project extracts, filters, and merges academic keywords from multiple diverse sources:

1. **Wikipedia Articles** - Academic terms from Wikipedia content
2. **ArXiv ML Papers** - Academic terms from machine learning papers on ArXiv
3. **PubMed/ArXiv Abstracts** - Biomedical and scientific abstracts from PubMed and ArXiv
4. **Processed PubMed Papers** - Full biomedical research papers from PubMed
5. **Scientific Papers Archive** - Diverse scientific papers across multiple disciplines

## Directory Structure

```
academic_keywords_extraction/
├── code/                      # Python scripts for data processing
├── data/                      # Data directories for different sources
│   ├── wikipedia/            # Wikipedia data
│   ├── pubmed/               # PubMed data
│   ├── arxiv/                # ArXiv data
│   └── scientific_papers/    # Scientific papers data
└── output/                   # Output files
    ├── wikipedia/            # Keywords extracted from Wikipedia
    ├── diverse/              # Keywords from diverse sources
    └── merged/               # Merged and filtered keywords
```

## Code Files

- `download_wikipedia.py` - Downloads Wikipedia articles
- `download_nltk_resources.py` - Downloads necessary NLTK resources
- `download_additional_nltk.py` - Downloads additional NLTK resources
- `fix_nltk_resources.py` - Fixes NLTK resource issues
- `extract_keywords.py` - Extracts keywords from Wikipedia articles
- `simple_keyword_extraction.py` - Simple version of keyword extraction
- `improved_keyword_extraction.py` - Improved version with better filtering
- `download_diverse_datasets.py` - Downloads diverse academic paper datasets
- `download_armanc_dataset.py` - Downloads armanc/scientific_papers dataset
- `extract_diverse_keywords.py` - Extracts keywords from diverse datasets
- `merge_and_filter_keywords.py` - Merges and filters all keywords

## Results

The project generated 294,071 unique academic keywords from diverse fields:

- **Medicine**: 2,074 occurrences (surgery, pathology, oncology, etc.)
- **Humanities**: 1,957 occurrences (history, literature, ethics, etc.)
- **Social Sciences**: 595 occurrences (education, psychology, economics, etc.)
- **Biology**: 555 occurrences (evolution, genetics, molecular biology, etc.)
- **Mathematics**: 476 occurrences (probability, statistics, algebra, etc.)
- **Computer Science**: 185 occurrences (machine learning, artificial intelligence, etc.)
- **Environmental**: 149 occurrences (pollution, climate change, etc.)
- **Physics**: 104 occurrences (quantum mechanics, relativity, etc.)
- **Engineering**: Various engineering disciplines

## Usage

1. Download the datasets:
   ```
   python code/download_wikipedia.py
   python code/download_diverse_datasets.py
   ```

2. Extract keywords:
   ```
   python code/extract_keywords.py
   python code/extract_diverse_keywords.py
   ```

3. Merge and filter keywords:
   ```
   python code/merge_and_filter_keywords.py
   ```

4. Use the keywords for Google Scholar search:
   - Use the more specific multi-word terms for targeted searches
   - Combine keywords with other search parameters (date ranges, authors, etc.)
   - Use quotation marks around multi-word terms for exact phrase matching
