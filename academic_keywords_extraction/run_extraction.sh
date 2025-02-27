#!/bin/bash

# Academic Keywords Extraction Pipeline
# This script runs the complete pipeline for extracting academic keywords

# Set up environment
echo "Setting up environment..."
mkdir -p data/{wikipedia,pubmed,arxiv,scientific_papers}
mkdir -p output/{wikipedia,diverse,merged}

# Step 1: Download NLTK resources
echo "Downloading NLTK resources..."
python code/download_nltk_resources.py
python code/download_additional_nltk.py
python code/fix_nltk_resources.py

# Step 2: Download Wikipedia data
echo "Downloading Wikipedia data..."
python code/download_wikipedia.py

# Step 3: Extract keywords from Wikipedia
echo "Extracting keywords from Wikipedia..."
python code/extract_keywords.py
# For improved extraction
python code/improved_keyword_extraction.py

# Step 4: Download diverse datasets
echo "Downloading diverse datasets..."
python code/download_diverse_datasets.py
python code/download_armanc_dataset.py

# Step 5: Extract keywords from diverse datasets
echo "Extracting keywords from diverse datasets..."
python code/extract_diverse_keywords.py

# Step 6: Merge and filter all keywords
echo "Merging and filtering keywords..."
python code/merge_and_filter_keywords.py

echo "Keyword extraction complete!"
echo "Results are available in the output directory"
