import nltk
import os
import sys

def download_nltk_resources():
    """Download all necessary NLTK resources."""
    resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'wordnet',
        'punkt_tab'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
    
    print("All resources downloaded!")

def fix_tagger_issue():
    """Fix the issue with averaged_perceptron_tagger_eng by creating a symlink."""
    nltk_data_dir = nltk.data.path[0]
    
    # Check if the directory exists
    tagger_dir = os.path.join(nltk_data_dir, 'taggers')
    if not os.path.exists(tagger_dir):
        print(f"Taggers directory not found at {tagger_dir}")
        return False
    
    # Check if the averaged_perceptron_tagger directory exists
    apt_dir = os.path.join(tagger_dir, 'averaged_perceptron_tagger')
    if not os.path.exists(apt_dir):
        print(f"averaged_perceptron_tagger directory not found at {apt_dir}")
        return False
    
    # Create the eng directory if it doesn't exist
    apt_eng_dir = os.path.join(tagger_dir, 'averaged_perceptron_tagger_eng')
    if not os.path.exists(apt_eng_dir):
        os.makedirs(apt_eng_dir, exist_ok=True)
        print(f"Created directory {apt_eng_dir}")
    
    # Copy files from averaged_perceptron_tagger to averaged_perceptron_tagger_eng
    for filename in os.listdir(apt_dir):
        src = os.path.join(apt_dir, filename)
        dst = os.path.join(apt_eng_dir, filename)
        if not os.path.exists(dst):
            if os.path.isfile(src):
                with open(src, 'rb') as f_src:
                    with open(dst, 'wb') as f_dst:
                        f_dst.write(f_src.read())
                print(f"Copied {src} to {dst}")
    
    print("Fixed tagger issue!")
    return True

if __name__ == "__main__":
    download_nltk_resources()
    fix_tagger_issue()
    print("NLTK resources setup complete!")
