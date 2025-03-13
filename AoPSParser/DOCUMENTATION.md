# AoPS Parser Documentation

## Files

- **final_enhanced_parser.py**: Main parser script that extracts problems, solutions, and images from AoPS WARC files
- **verify_improved_extraction.py**: Verification script to check the quality of extraction
- **create_improved_sample.py**: Script to create sample output files

## Output Structure

```json
[
  {
    "url": "https://artofproblemsolving.com/wiki/index.php/...",
    "page_url": "https://artofproblemsolving.com/wiki/index.php/...",
    "title": "Art of Problem Solving",
    "problem_list": [
      {
        "problem": "Problem text with properly spaced LaTeX expressions like $ x $ and image placeholders like [IMAGE_1]",
        "image_list": ["https://artofproblemsolving.com/image1.png"],
        "solution_list": [
          {
            "title": "Solution 1 (Descriptive Title)",
            "solution": "Solution text with properly spaced LaTeX expressions and image placeholders",
            "image_list": ["https://artofproblemsolving.com/image2.png"]
          }
        ]
      }
    ]
  }
]
```

## Requirements

- Python 3.8+
- warcio
- beautifulsoup4
- requests

## Installation

```bash
pip install warcio beautifulsoup4 requests
```
