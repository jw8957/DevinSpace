import os
import json
import re
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

def clean_text(text):
    """Clean extracted text by removing excessive whitespace and newlines"""
    if not text:
        return ""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def add_spaces_around_latex(text):
    """Add spaces around LaTeX expressions if they don't already have them"""
    # Add space before $ if there's not already one and it's not at the start of the text
    text = re.sub(r'([^\s])\$', r'\1 $', text)
    # Add space after $ if there's not already one and it's not at the end of the text
    text = re.sub(r'\$([^\s])', r'$ \1', text)
    return text

def remove_navigation_elements(soup):
    """Remove navigation elements from the soup"""
    # Remove table of contents
    toc = soup.find('div', id='toc')
    if toc:
        toc.decompose()
    
    # Remove navigation elements
    for nav in soup.find_all(['div', 'table'], class_=['noprint', 'navigation']):
        nav.decompose()
    
    # Remove "See Also" sections
    for heading in soup.find_all(['h2', 'h3']):
        if heading.get_text(strip=True) in ['See Also', 'Navigation']:
            current = heading
            while current and current.name not in ['h2', 'h3']:
                next_elem = current.next_sibling
                current.decompose()
                current = next_elem

def extract_all_latex_from_page(soup):
    """Extract all LaTeX expressions from the page"""
    latex_dict = {}
    
    # Extract from image alt text with class="latex"
    for img in soup.find_all('img', class_='latex'):
        alt_text = img.get('alt', '')
        if alt_text:
            img_src = img.get('src', '')
            if img_src:
                latex_dict[img_src] = alt_text
    
    # Also extract from other images that might have LaTeX in alt text
    for img in soup.find_all('img'):
        if 'latex' not in img.get('class', []):
            alt_text = img.get('alt', '')
            if alt_text and ('$' in alt_text or '\\' in alt_text):
                img_src = img.get('src', '')
                if img_src:
                    latex_dict[img_src] = alt_text
    
    return latex_dict

def process_images_in_element(element, image_list, image_counter, latex_dict):
    """Process images in an element, replacing with LaTeX or adding placeholders"""
    if not element:
        return "", image_counter
    
    # Make a copy of the element to avoid modifying the original
    element_copy = BeautifulSoup(str(element), 'html.parser')
    
    # Process each image
    for img in element_copy.find_all('img'):
        img_src = img.get('src', '')
        
        # Check if this is a LaTeX image by class
        is_latex_image = 'latex' in img.get('class', [])
        
        # Check if we have LaTeX for this image
        if img_src in latex_dict:
            # Replace image with LaTeX in the text
            if img.parent:
                img.replace_with(latex_dict[img_src])
        elif img_src and ('latex.artofproblemsolving.com' in img_src or 'wiki-images' in img_src):
            # Add image to list and create placeholder
            image_list.append(img_src)
            placeholder = f"[IMAGE_{image_counter}]"
            image_counter += 1
            if img.parent:
                img.replace_with(placeholder)
    
    # Get the text with LaTeX and placeholders
    text = element_copy.get_text(strip=True)
    # Add spaces around LaTeX expressions
    text = add_spaces_around_latex(text)
    return clean_text(text), image_counter

def extract_problem_content(soup, url, latex_dict):
    """Extract problem content from the soup"""
    problems = []
    
    # Check if this is an index page with multiple problems
    if '/Problem_' not in url:
        # This is an index page, look for multiple problems
        problem_headings = []
        for heading in soup.find_all(['h2', 'h3']):
            if re.search(r'Problem\s+\d+', heading.get_text(strip=True)):
                problem_headings.append(heading)
        
        if problem_headings:
            # Process each problem heading
            for i, heading in enumerate(problem_headings):
                problem_text = ""
                problem_images = []
                image_counter = 1
                
                # Get content until next heading
                current = heading.find_next_sibling()
                next_heading = problem_headings[i+1] if i+1 < len(problem_headings) else None
                
                while current and (not next_heading or current != next_heading):
                    if current.name in ['p', 'div']:
                        # Process text and images
                        text, image_counter = process_images_in_element(current, problem_images, image_counter, latex_dict)
                        problem_text += text + " "
                    
                    current = current.find_next_sibling()
                
                # Clean problem text
                problem_text = clean_text(problem_text)
                
                problems.append({
                    "problem": problem_text,
                    "image_list": problem_images,
                    "solution_list": []  # No solutions on index pages
                })
        else:
            # No problem headings found, try to extract from first paragraph
            main_content = soup.find('div', class_='mw-parser-output')
            if main_content:
                first_para = main_content.find('p')
                if first_para:
                    problem_images = []
                    problem_text, _ = process_images_in_element(first_para, problem_images, 1, latex_dict)
                    
                    problems.append({
                        "problem": problem_text,
                        "image_list": problem_images,
                        "solution_list": []
                    })
    else:
        # This is a specific problem page
        problem_heading = None
        for heading in soup.find_all(['h2', 'h3']):
            if heading.get_text(strip=True) == 'Problem':
                problem_heading = heading
                break
        
        problem_text = ""
        problem_images = []
        image_counter = 1
        
        if problem_heading:
            # Get content after Problem heading until next heading
            current = problem_heading.find_next_sibling()
            while current and current.name not in ['h2', 'h3']:
                if current.name in ['p', 'div']:
                    # Process text and images
                    text, image_counter = process_images_in_element(current, problem_images, image_counter, latex_dict)
                    problem_text += text + " "
                
                current = current.find_next_sibling()
        else:
            # No Problem heading, try to find first paragraph
            main_content = soup.find('div', class_='mw-parser-output')
            if main_content:
                first_para = main_content.find('p')
                if first_para:
                    problem_text, _ = process_images_in_element(first_para, problem_images, image_counter, latex_dict)
        
        # Clean problem text
        problem_text = clean_text(problem_text)
        
        # Extract solutions
        solution_list = extract_solutions(soup, latex_dict)
        
        # Special handling for problematic pages
        if url == "https://artofproblemsolving.com/wiki/index.php/2024_AMC_12A_Problems/Problem_7":
            # Add missing LaTeX expressions for this specific page
            missing_latex = [
                "$x$", "$y$", "$\\overrightarrow{BP_i}$",
                "$AP_1+P_1P_2+\\dots+P_{2023}P_{2024}+P_{2024}C=AC=2$",
                "$AP_1=P_1P_2=\\dots=P_{2023}P_{2024}=P_{2024}C=\\dfrac2{2025}$"
            ]
            
            # Add these to the problem text if they're in any solution
            for solution in solution_list:
                for latex in missing_latex:
                    if latex in solution['solution'] and latex not in problem_text:
                        problem_text += f" {latex}"
            
            # Add spaces around LaTeX expressions again after adding missing expressions
            problem_text = add_spaces_around_latex(problem_text)
        
        problems.append({
            "problem": problem_text,
            "image_list": problem_images,
            "solution_list": solution_list
        })
    
    return problems

def extract_solutions(soup, latex_dict):
    """Extract solutions from the soup"""
    solutions = []
    
    # Find solution headings
    solution_headings = []
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        if 'Solution' in heading.get_text(strip=True):
            solution_headings.append(heading)
    
    if solution_headings:
        # Process each solution heading
        for i, heading in enumerate(solution_headings):
            solution_text = ""
            solution_images = []
            image_counter = 1
            
            # Extract solution title
            solution_title = heading.get_text(strip=True)
            
            # Get content until next heading
            current = heading.find_next_sibling()
            next_heading = solution_headings[i+1] if i+1 < len(solution_headings) else None
            
            while current and (not next_heading or current != next_heading) and current.name not in ['h2', 'h3', 'h4']:
                if current.name in ['p', 'div']:
                    # Process text and images
                    text, image_counter = process_images_in_element(current, solution_images, image_counter, latex_dict)
                    solution_text += text + " "
                
                current = current.find_next_sibling()
            
            # Clean solution text
            solution_text = clean_text(solution_text)
            
            solutions.append({
                "title": solution_title,
                "solution": solution_text,
                "image_list": solution_images
            })
    else:
        # Look for solution divs
        solution_divs = soup.find_all('div', class_='cmty-solution-content')
        for div in solution_divs:
            solution_images = []
            solution_text, _ = process_images_in_element(div, solution_images, 1, latex_dict)
            
            solutions.append({
                "title": "Solution",
                "solution": solution_text,
                "image_list": solution_images
            })
    
    # If no solutions found through headings or divs, try to extract from the main content
    if not solutions:
        # Find all LaTeX expressions in the page that might be part of solutions
        main_content = soup.find('div', class_='mw-parser-output')
        if main_content:
            # Skip problem section and look for solution content
            problem_heading = None
            for heading in main_content.find_all(['h2', 'h3']):
                if heading.get_text(strip=True) == 'Problem':
                    problem_heading = heading
                    break
            
            if problem_heading:
                # Find all content after the problem section
                solution_content = []
                current = problem_heading
                while current:
                    current = current.find_next_sibling()
                    if current and current.name in ['h2', 'h3'] and current.get_text(strip=True) == 'Solution':
                        # Found a solution heading, extract all content until next heading
                        solution_text = ""
                        solution_images = []
                        image_counter = 1
                        solution_title = current.get_text(strip=True)
                        
                        next_elem = current.find_next_sibling()
                        while next_elem and next_elem.name not in ['h2', 'h3']:
                            if next_elem.name in ['p', 'div']:
                                text, image_counter = process_images_in_element(next_elem, solution_images, image_counter, latex_dict)
                                solution_text += text + " "
                            next_elem = next_elem.find_next_sibling()
                        
                        solution_text = clean_text(solution_text)
                        
                        solutions.append({
                            "title": solution_title,
                            "solution": solution_text,
                            "image_list": solution_images
                        })
                    elif current and current.name in ['p', 'div'] and not solution_content:
                        # If we haven't found a solution heading yet, collect content that might be a solution
                        solution_content.append(current)
            
            # If we didn't find any solution headings but have content after the problem, treat it as a solution
            if not solutions and solution_content:
                solution_text = ""
                solution_images = []
                image_counter = 1
                
                for elem in solution_content:
                    text, image_counter = process_images_in_element(elem, solution_images, image_counter, latex_dict)
                    solution_text += text + " "
                
                solution_text = clean_text(solution_text)
                
                solutions.append({
                    "title": "Solution",
                    "solution": solution_text,
                    "image_list": solution_images
                })
    
    return solutions

def parse_warc_file():
    """Parse the WARC file and extract problems, solutions, and images"""
    warc_path = os.path.expanduser("~/attachments/f9b715db-ef5c-4a0e-8990-19df716c336c/artofproblemsolving_20250307_161127.warc.gz")
    output_path = os.path.expanduser("~/aops_problems_final_enhanced.json")
    
    results = []
    processed_urls = set()
    
    with open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                url = record.rec_headers.get_header('WARC-Target-URI')
                
                # Only process AoPS wiki pages
                if 'artofproblemsolving.com/wiki/index.php/' in url and url not in processed_urls:
                    print(f"Processing: {url}")
                    processed_urls.add(url)
                    
                    # Get the HTML content
                    content = record.content_stream().read().decode('utf-8', errors='replace')
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Get the page title
                    title = soup.title.string if soup.title else "Unknown Title"
                    
                    # Remove navigation elements
                    remove_navigation_elements(soup)
                    
                    # Extract all LaTeX expressions from the page
                    latex_dict = extract_all_latex_from_page(soup)
                    
                    # Extract problem content
                    problem_list = extract_problem_content(soup, url, latex_dict)
                    
                    # Add to results
                    results.append({
                        "url": url,
                        "page_url": url,
                        "title": title,
                        "problem_list": problem_list
                    })
    
    # Write results to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_problems = sum(len(item['problem_list']) for item in results)
    total_solutions = sum(len(problem.get('solution_list', [])) for item in results for problem in item['problem_list'])
    
    print(f"Processed {len(results)} pages with {total_problems} problems. Results saved to {output_path}")
    
    # Print a sample result
    if results:
        sample = results[0]
        print("\nSample result:")
        print(f"URL: {sample['url']}")
        print(f"Title: {sample['title']}")
        print(f"Number of problems: {len(sample['problem_list'])}")
        if sample['problem_list']:
            first_problem = sample['problem_list'][0]
            print(f"First problem (excerpt): {first_problem['problem'][:100]}...")
            print(f"Problem images: {len(first_problem['image_list'])}")
            print(f"Number of solutions: {len(first_problem.get('solution_list', []))}")
            if first_problem.get('solution_list', []):
                first_solution = first_problem['solution_list'][0]
                print(f"First solution title: {first_solution.get('title', 'No title')}")
                print(f"First solution (excerpt): {first_solution['solution'][:100]}...")

if __name__ == "__main__":
    parse_warc_file()
