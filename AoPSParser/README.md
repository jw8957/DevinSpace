# AoPS WARC Parser

A parser for extracting structured data from Art of Problem Solving WARC files.

## Features

- Extracts problems and solutions from AoPS pages
- Preserves LaTeX expressions with proper spacing
- Extracts solution titles including descriptive text
- Creates image placeholders for non-LaTeX images
- Maintains nested structure with problem_list and solution_list

## Usage

```python
python final_enhanced_parser.py
```

The parser will process the WARC file and output a JSON file with the extracted data.
