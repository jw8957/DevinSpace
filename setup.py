from setuptools import setup, find_packages

setup(
    name="RephraseModel",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'scikit-learn',
        'numpy',
        'spacy',
        'beautifulsoup4',
        'tqdm',
        'requests',
        'pandas'
    ],
    python_requires='>=3.8',
)
