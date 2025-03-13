import json
import os

def create_improved_sample():
    """Create a sample output file with representative records"""
    json_path = os.path.expanduser("~/aops_problems_final_enhanced.json")
    sample_path = os.path.expanduser("~/improved_sample_output.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take a representative sample: first 2 records, plus the problematic pages
    sample_data = data[:2]
    
    # Add the specific problematic pages
    problematic_urls = [
        "https://artofproblemsolving.com/wiki/index.php/2024_AMC_12A_Problems/Problem_7",
        "https://artofproblemsolving.com/wiki/index.php/2024_AMC_12A_Problems/Problem_25"
    ]
    
    for url in problematic_urls:
        page_data = next((item for item in data if item['url'] == url), None)
        if page_data:
            sample_data.append(page_data)
    
    # Write sample to JSON file
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Improved sample output created with {len(sample_data)} records. Saved to {sample_path}")
    
    # Calculate statistics
    total_problems = sum(len(item['problem_list']) for item in sample_data)
    total_solutions = sum(len(problem.get('solution_list', [])) for item in sample_data for problem in item['problem_list'])
    total_latex_expressions = 0
    total_image_placeholders = 0
    
    # Count LaTeX expressions and image placeholders
    for item in sample_data:
        for problem in item['problem_list']:
            problem_text = problem['problem']
            if '$' in problem_text:
                total_latex_expressions += problem_text.count('$') // 2  # Divide by 2 for pairs
            
            # Count image placeholders
            if '[IMAGE_' in problem_text:
                total_image_placeholders += problem_text.count('[IMAGE_')
            
            for solution in problem.get('solution_list', []):
                solution_text = solution['solution']
                if '$' in solution_text:
                    total_latex_expressions += solution_text.count('$') // 2
                
                # Count image placeholders
                if '[IMAGE_' in solution_text:
                    total_image_placeholders += solution_text.count('[IMAGE_')
    
    print("\nSample statistics:")
    print(f"Total problems: {total_problems}")
    print(f"Total solutions: {total_solutions}")
    print(f"Total LaTeX expressions: {total_latex_expressions}")
    print(f"Total image placeholders: {total_image_placeholders}")
    
    # Print sample structure
    print("\nSample structure:")
    for i, item in enumerate(sample_data):
        print(f"\nRecord {i+1}:")
        print(f"URL: {item['url']}")
        print(f"Title: {item['title']}")
        print(f"Number of problems: {len(item['problem_list'])}")
        
        if item['problem_list']:
            first_problem = item['problem_list'][0]
            print(f"First problem (excerpt): {first_problem['problem'][:100]}...")
            
            # Count LaTeX expressions
            latex_count = first_problem['problem'].count('$') // 2
            print(f"LaTeX expressions: {latex_count}")
            
            # Count image placeholders
            placeholder_count = first_problem['problem'].count('[IMAGE_')
            print(f"Image placeholders: {placeholder_count}")
            
            print(f"Problem images: {len(first_problem['image_list'])}")
            print(f"Number of solutions: {len(first_problem.get('solution_list', []))}")
            
            if first_problem.get('solution_list', []):
                first_solution = first_problem['solution_list'][0]
                print(f"First solution title: {first_solution.get('title', 'No title')}")
                print(f"First solution (excerpt): {first_solution['solution'][:100]}...")

if __name__ == "__main__":
    create_improved_sample()
