import json
import os
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator

def verify_improved_extraction():
    """Verify the improved extraction with spaces around LaTeX and solution titles"""
    json_path = os.path.expanduser("~/aops_problems_final_enhanced.json")
    
    # Load the extracted data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} pages from {json_path}")
    
    # Check specific problematic pages
    check_specific_page(data, "https://artofproblemsolving.com/wiki/index.php/2024_AMC_12A_Problems/Problem_25")
    check_specific_page(data, "https://artofproblemsolving.com/wiki/index.php/2024_AMC_12A_Problems/Problem_7")
    
    # Check a few random pages
    random_pages = [
        "https://artofproblemsolving.com/wiki/index.php/2024_AMC_8_Problems/Problem_1",
        "https://artofproblemsolving.com/wiki/index.php/2024_AMC_10A_Problems/Problem_10"
    ]
    
    for url in random_pages:
        check_specific_page(data, url)
    
    # Verify LaTeX spacing across all pages
    verify_latex_spacing(data)
    
    # Verify solution titles across all pages
    verify_solution_titles(data)

def check_specific_page(data, url_to_check):
    """Check a specific page in detail"""
    page_data = next((item for item in data if item['url'] == url_to_check), None)
    
    if not page_data:
        print(f"Page not found in extraction results: {url_to_check}")
        return
    
    print(f"\n=== CHECKING SPECIFIC PAGE: {url_to_check} ===")
    print(f"Title: {page_data['title']}")
    print(f"Number of problems: {len(page_data['problem_list'])}")
    
    if page_data['problem_list']:
        problem = page_data['problem_list'][0]
        print(f"\nProblem text (excerpt):")
        print(problem['problem'][:300] + "..." if len(problem['problem']) > 300 else problem['problem'])
        
        # Check for proper spacing around LaTeX expressions
        check_latex_spacing(problem['problem'])
        
        # Count LaTeX expressions
        latex_count = problem['problem'].count('$') // 2
        print(f"\nLaTeX expressions: {latex_count}")
        
        # Count image placeholders
        placeholder_count = problem['problem'].count('[IMAGE_')
        print(f"Image placeholders: {placeholder_count}")
        
        print(f"Problem images: {len(problem['image_list'])}")
        
        # Check solutions
        print(f"\nNumber of solutions: {len(problem.get('solution_list', []))}")
        
        for i, solution in enumerate(problem.get('solution_list', [])[:3]):  # Show first 3 solutions
            print(f"\nSolution {i+1}:")
            print(f"Title: {solution.get('title', 'No title')}")
            print(f"Text (excerpt): {solution['solution'][:200]}..." if len(solution['solution']) > 200 else solution['solution'])
            
            # Check for proper spacing around LaTeX expressions
            check_latex_spacing(solution['solution'])
            
            # Count LaTeX expressions
            latex_count = solution['solution'].count('$') // 2
            print(f"LaTeX expressions: {latex_count}")
            
            # Count image placeholders
            placeholder_count = solution['solution'].count('[IMAGE_')
            print(f"Image placeholders: {placeholder_count}")
            
            # Check image list
            if solution['image_list']:
                print(f"Solution images: {len(solution['image_list'])}")
            else:
                print("No images in solution image_list")

def check_latex_spacing(text):
    """Check if LaTeX expressions have proper spacing around them"""
    if '$' not in text:
        return
    
    # Find all LaTeX expressions
    latex_expressions = []
    in_latex = False
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '$':
            if not in_latex:
                start_idx = i
                in_latex = True
            else:
                latex_expressions.append(text[start_idx:i+1])
                in_latex = False
    
    # Check spacing around LaTeX expressions
    improper_spacing = []
    for latex in latex_expressions:
        # Check if there's a space before the expression (unless it's at the start of the text)
        if latex in text and text.index(latex) > 0:
            before_char = text[text.index(latex) - 1]
            if before_char != ' ' and before_char != '\n':
                improper_spacing.append(f"No space before: {latex}")
        
        # Check if there's a space after the expression (unless it's at the end of the text)
        if latex in text and text.index(latex) + len(latex) < len(text):
            after_char = text[text.index(latex) + len(latex)]
            if after_char != ' ' and after_char != '\n' and after_char != '.':
                improper_spacing.append(f"No space after: {latex}")
    
    if improper_spacing:
        print("\nImproper spacing around LaTeX expressions:")
        for issue in improper_spacing[:5]:  # Show first 5 issues
            print(f"  {issue}")
        if len(improper_spacing) > 5:
            print(f"  ... and {len(improper_spacing) - 5} more")
    else:
        print("\nAll LaTeX expressions have proper spacing")

def verify_latex_spacing(data):
    """Verify LaTeX spacing across all pages"""
    total_problems = 0
    problems_with_improper_spacing = 0
    
    for page in data:
        for problem in page['problem_list']:
            total_problems += 1
            
            # Check problem text
            if has_improper_latex_spacing(problem['problem']):
                problems_with_improper_spacing += 1
            
            # Check solutions
            for solution in problem.get('solution_list', []):
                if has_improper_latex_spacing(solution['solution']):
                    problems_with_improper_spacing += 1
    
    print(f"\n=== LATEX SPACING VERIFICATION ===")
    print(f"Total problems and solutions checked: {total_problems}")
    print(f"Problems/solutions with improper spacing: {problems_with_improper_spacing}")
    print(f"Percentage with proper spacing: {100 - (problems_with_improper_spacing / total_problems * 100):.2f}%")

def has_improper_latex_spacing(text):
    """Check if text has improper spacing around LaTeX expressions"""
    if '$' not in text:
        return False
    
    # Find all LaTeX expressions
    latex_expressions = []
    in_latex = False
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '$':
            if not in_latex:
                start_idx = i
                in_latex = True
            else:
                latex_expressions.append(text[start_idx:i+1])
                in_latex = False
    
    # Check spacing around LaTeX expressions
    for latex in latex_expressions:
        # Check if there's a space before the expression (unless it's at the start of the text)
        if latex in text and text.index(latex) > 0:
            before_char = text[text.index(latex) - 1]
            if before_char != ' ' and before_char != '\n':
                return True
        
        # Check if there's a space after the expression (unless it's at the end of the text)
        if latex in text and text.index(latex) + len(latex) < len(text):
            after_char = text[text.index(latex) + len(latex)]
            if after_char != ' ' and after_char != '\n' and after_char != '.':
                return True
    
    return False

def verify_solution_titles(data):
    """Verify solution titles across all pages"""
    total_solutions = 0
    solutions_with_titles = 0
    
    for page in data:
        for problem in page['problem_list']:
            for solution in problem.get('solution_list', []):
                total_solutions += 1
                if solution.get('title') and solution['title'] != "Solution":
                    solutions_with_titles += 1
    
    print(f"\n=== SOLUTION TITLE VERIFICATION ===")
    print(f"Total solutions: {total_solutions}")
    print(f"Solutions with specific titles: {solutions_with_titles}")
    print(f"Percentage with specific titles: {solutions_with_titles / total_solutions * 100:.2f}%")

if __name__ == "__main__":
    verify_improved_extraction()
