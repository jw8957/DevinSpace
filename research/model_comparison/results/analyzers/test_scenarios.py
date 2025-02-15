import json
import torch
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestCase:
    name: str
    description: str
    input_text: str
    expected_output: bool
    category: str
    language: str
    complexity: str

class TestScenarioManager:
    def __init__(self, output_dir='../results'):
        self.output_dir = output_dir
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load predefined test cases"""
        return [
            TestCase(
                name="navigation_menu",
                description="Navigation menu with multiple links",
                input_text="Home About Products Services Contact",
                expected_output=False,
                category="navigation",
                language="en",
                complexity="simple"
            ),
            TestCase(
                name="article_content",
                description="Main article content",
                input_text="This is the main content of the article...",
                expected_output=True,
                category="content",
                language="en",
                complexity="moderate"
            ),
            TestCase(
                name="social_widgets",
                description="Social media sharing buttons",
                input_text="Share on Facebook Twitter LinkedIn",
                expected_output=False,
                category="social",
                language="en",
                complexity="simple"
            ),
            TestCase(
                name="chinese_navigation",
                description="Chinese navigation menu",
                input_text="首页 关于我们 产品 服务 联系方式",
                expected_output=False,
                category="navigation",
                language="zh",
                complexity="simple"
            ),
            TestCase(
                name="mixed_language",
                description="Mixed language content",
                input_text="Product Features 产品特点",
                expected_output=True,
                category="content",
                language="mixed",
                complexity="moderate"
            )
        ]
    
    def run_test_cases(self, model, tokenizer, device) -> Dict[str, Any]:
        """Run all test cases and collect results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'cases': []
        }
        
        for case in self.test_cases:
            prediction = self._run_single_test(model, tokenizer, device, case)
            passed = prediction == case.expected_output
            
            results['cases'].append({
                'name': case.name,
                'passed': passed,
                'expected': case.expected_output,
                'predicted': prediction,
                'category': case.category,
                'language': case.language
            })
            
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def _run_single_test(self, model, tokenizer, device, case: TestCase) -> bool:
        """Run a single test case"""
        if tokenizer is None:
            return False  # Skip test if tokenizer not available
            
        try:
            inputs = tokenizer(case.input_text, 
                             return_tensors='pt',
                             padding=True,
                             truncation=True)
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'].to(device),
                          inputs['attention_mask'].to(device))
            prediction = torch.argmax(outputs, dim=-1)[0].cpu().item()
        
        return bool(prediction)
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed test report"""
        report = f"# Model Test Report\nGenerated: {results['timestamp']}\n\n"
        
        # Overall statistics
        report += "## Summary\n"
        report += f"- Total Test Cases: {results['total_cases']}\n"
        report += f"- Passed: {results['passed']}\n"
        report += f"- Failed: {results['failed']}\n"
        report += f"- Success Rate: {results['passed']/results['total_cases']*100:.1f}%\n\n"
        
        # Category breakdown
        categories = {}
        for case in results['cases']:
            cat = case['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if case['passed']:
                categories[cat]['passed'] += 1
        
        report += "## Performance by Category\n"
        for cat, stats in categories.items():
            success_rate = stats['passed'] / stats['total'] * 100
            report += f"- {cat.title()}: {success_rate:.1f}% "
            report += f"({stats['passed']}/{stats['total']})\n"
        
        return report
