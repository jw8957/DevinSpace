#!/usr/bin/env python3
"""
Enhanced Chinese Research Query Pipeline

This script implements all five stages from the paper:
"Researchy Questions: A Dataset of Multi-Perspective, Decompositional Questions for LLM Web Agents"

Processing a large number of questions to generate around 500 research queries
covering different topics.

1. Stage 1: Mining Search Logs (using enhanced Zhihu questions)
2. Stage 2: Factoid Classifier
3. Stage 3: Decompositional Classifier
4. Stage 4: Deduplication
5. Stage 5: Final Filtering
"""

import json
import os
import re
import random
from datetime import datetime
from collections import Counter
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_query_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_query_pipeline")

class EnhancedQueryPipeline:
    """
    Enhanced pipeline for processing Chinese research queries through all five stages
    from the paper methodology, targeting around 500 final queries
    """
    
    def __init__(self, input_file='enhanced_zhihu_data/enhanced_zhihu_questions.json', 
                 output_dir='enhanced_pipeline_output'):
        """Initialize the pipeline with input and output paths"""
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Statistics to track
        self.stats = {
            "stage1_count": 0,
            "stage2_count": 0,
            "stage3_count": 0,
            "stage4_count": 0,
            "stage5_count": 0,
            "final_count": 0,
            "topic_distribution": {}
        }
    
    def run_pipeline(self, target_count=500):
        """Run the full pipeline through all five stages"""
        logger.info(f"Starting the Enhanced Chinese Research Query Pipeline (target: {target_count} queries)")
        
        # Stage 1: Mining Search Logs (using enhanced Zhihu questions)
        questions = self.stage1_mining_search_logs()
        self.save_stage_output(questions, "stage1_output.json")
        
        # Stage 2: Factoid Classifier
        non_factoid_questions = self.stage2_factoid_classifier(questions)
        self.save_stage_output(non_factoid_questions, "stage2_output.json")
        
        # Stage 3: Decompositional Classifier
        decomposable_questions = self.stage3_decompositional_classifier(non_factoid_questions)
        self.save_stage_output(decomposable_questions, "stage3_output.json")
        
        # Stage 4: Deduplication
        deduplicated_questions = self.stage4_deduplication(decomposable_questions)
        self.save_stage_output(deduplicated_questions, "stage4_output.json")
        
        # Stage 5: Final Filtering
        final_questions = self.stage5_final_filtering(deduplicated_questions)
        self.save_stage_output(final_questions, "stage5_output.json")
        
        # Generate research queries from the final questions
        research_queries = self.generate_research_queries(final_questions, target_count)
        self.save_stage_output(research_queries, "final_research_queries.json")
        
        # Save just the query strings
        query_strings = [query["research_query"] for query in research_queries]
        with open(os.path.join(self.output_dir, "query_strings.json"), 'w', encoding='utf-8') as f:
            json.dump(query_strings, f, ensure_ascii=False, indent=2)
        
        # Log statistics
        logger.info("Pipeline Statistics:")
        for stage, count in self.stats.items():
            if isinstance(count, dict):
                logger.info(f"  {stage}: {len(count)} categories")
            else:
                logger.info(f"  {stage}: {count}")
        
        return research_queries
    
    def stage1_mining_search_logs(self):
        """
        Stage 1: Mining Search Logs
        
        In the paper, they used Bing search logs.
        We're using enhanced Zhihu questions as our source data.
        """
        logger.info("Stage 1: Mining Search Logs (using enhanced Zhihu questions)")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            logger.info(f"Loaded {len(questions)} questions from {self.input_file}")
            
            # Track topic distribution
            topic_counts = {}
            for q in questions:
                topic = q.get('topic', 'Unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            logger.info(f"Questions cover {len(topic_counts)} different topics")
            self.stats["stage1_count"] = len(questions)
            self.stats["topic_distribution"] = topic_counts
            return questions
        except Exception as e:
            logger.error(f"Error in Stage 1: {str(e)}")
            return []
    
    def stage2_factoid_classifier(self, questions):
        """
        Stage 2: Factoid Classifier
        
        Filter out factoid questions (questions with simple, direct answers).
        Non-factoid questions require longer, more complex answers.
        """
        logger.info("Stage 2: Factoid Classifier")
        
        non_factoid_questions = []
        
        for question in questions:
            if not self.is_factoid(question["text"]):
                non_factoid_questions.append(question)
        
        logger.info(f"Filtered out factoid questions. {len(non_factoid_questions)} non-factoid questions remain.")
        self.stats["stage2_count"] = len(non_factoid_questions)
        return non_factoid_questions
    
    def is_factoid(self, question_text):
        """
        Determine if a question is factoid (simple, single-answer)
        
        Factoid questions typically:
        - Are shorter
        - Ask for specific facts, dates, names, etc.
        - Often start with who, what, when, where, which
        """
        # Check length (factoid questions tend to be shorter)
        if len(question_text) < 10:
            return True
        
        # Check for factoid indicators in Chinese
        factoid_indicators = [
            "是谁", "是什么", "在哪里", "多少", "哪一个", "何时", 
            "为何", "怎么", "如何做", "怎样做"
        ]
        
        for indicator in factoid_indicators:
            if indicator in question_text:
                return True
        
        # Check for question patterns that suggest factoid
        factoid_patterns = [
            r"^谁.*\?$",  # Who questions
            r"^什么.*\?$",  # What questions
            r"^哪.*\?$",  # Which questions
            r"^何时.*\?$",  # When questions
            r"^在哪里.*\?$",  # Where questions
        ]
        
        for pattern in factoid_patterns:
            if re.search(pattern, question_text):
                return True
        
        return False
    
    def stage3_decompositional_classifier(self, questions):
        """
        Stage 3: Decompositional Classifier
        
        Filter for questions that can be decomposed into sub-questions.
        These are more suitable for research queries.
        """
        logger.info("Stage 3: Decompositional Classifier")
        
        decomposable_questions = []
        
        for question in questions:
            if self.is_decomposable(question["text"]):
                decomposable_questions.append(question)
        
        logger.info(f"Filtered for decomposable questions. {len(decomposable_questions)} decomposable questions remain.")
        self.stats["stage3_count"] = len(decomposable_questions)
        return decomposable_questions
    
    def is_decomposable(self, question_text):
        """
        Determine if a question is decomposable into sub-questions
        
        Decomposable questions typically:
        - Are more complex
        - Cover multiple aspects or dimensions
        - Require analysis from different angles
        """
        # Check length (decomposable questions tend to be longer)
        if len(question_text) < 15:
            return False
        
        # Check for decomposable indicators in Chinese
        decomposable_indicators = [
            "分析", "比较", "评价", "如何看待", "未来趋势", "发展", 
            "影响", "关系", "优缺点", "多方面", "不同角度"
        ]
        
        indicator_count = sum(1 for indicator in decomposable_indicators if indicator in question_text)
        if indicator_count >= 1:
            return True
        
        return False
    
    def stage4_deduplication(self, questions, similarity_threshold=0.7):
        """
        Stage 4: Deduplication
        
        Remove similar questions based on text similarity.
        """
        logger.info("Stage 4: Deduplication")
        
        if not questions:
            logger.warning("No questions to deduplicate")
            self.stats["stage4_count"] = 0
            return []
        
        # Ensure we maintain topic diversity
        questions_by_topic = {}
        for q in questions:
            topic = q.get('topic', 'Unknown')
            if topic not in questions_by_topic:
                questions_by_topic[topic] = []
            questions_by_topic[topic].append(q)
        
        # Deduplicate within each topic
        deduplicated = []
        for topic, topic_questions in questions_by_topic.items():
            topic_deduplicated = self.deduplicate_questions(topic_questions, similarity_threshold)
            deduplicated.extend(topic_deduplicated)
            logger.info(f"Topic '{topic}': {len(topic_questions)} -> {len(topic_deduplicated)} questions after deduplication")
        
        logger.info(f"Deduplicated questions. {len(deduplicated)} unique questions remain across {len(questions_by_topic)} topics.")
        self.stats["stage4_count"] = len(deduplicated)
        return deduplicated
    
    def deduplicate_questions(self, questions, similarity_threshold):
        """Deduplicate a list of questions based on text similarity"""
        if not questions:
            return []
        
        deduplicated = [questions[0]]
        
        for i in range(1, len(questions)):
            is_duplicate = False
            for existing in deduplicated:
                similarity = self.compute_similarity(questions[i]["text"], existing["text"])
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(questions[i])
        
        return deduplicated
    
    def compute_similarity(self, text1, text2):
        """
        Compute a simple similarity score between two texts
        using character-level Jaccard similarity
        """
        # Convert to sets of characters
        set1 = set(text1)
        set2 = set(text2)
        
        # Compute Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def stage5_final_filtering(self, questions):
        """
        Stage 5: Final Filtering
        
        Apply final quality filters to ensure questions meet all criteria.
        In the paper, they used GPT-4 for this stage.
        We're using heuristics to simulate this.
        """
        logger.info("Stage 5: Final Filtering")
        
        filtered_questions = []
        
        for question in questions:
            text = question["text"]
            
            # Check if it's in Chinese
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
            if not chinese_pattern.search(text):
                continue
            
            # Check if it's well-formed (ends with a question mark)
            if not (text.endswith("?") or text.endswith("？")):
                # Add a question mark if missing
                text = text + "？"
                question["text"] = text
            
            # Check if it's complex enough (at least 10 characters)
            if len(text) < 10:
                continue
            
            # Check if it contains at least one topic-related keyword
            topic_keywords = [
                "经济", "政治", "社会", "文化", "科技", "教育", "医疗", "环保",
                "金融", "投资", "创业", "职场", "健康", "心理", "历史", "艺术",
                "体育", "旅游", "互联网", "区块链", "人工智能", "大数据"
            ]
            
            has_topic = any(keyword in text for keyword in topic_keywords)
            if not has_topic:
                continue
            
            filtered_questions.append(question)
        
        logger.info(f"Applied final filtering. {len(filtered_questions)} high-quality questions remain.")
        self.stats["stage5_count"] = len(filtered_questions)
        return filtered_questions
    
    def generate_research_queries(self, questions, target_count=500):
        """
        Generate research queries from the filtered questions
        
        This transforms the filtered questions into complex research queries
        following the format described in the paper.
        """
        logger.info(f"Generating research queries (target: {target_count})")
        
        # If we don't have enough questions, we'll need to generate multiple queries per question
        queries_per_question = max(1, target_count // len(questions) + 1)
        logger.info(f"Generating approximately {queries_per_question} queries per question")
        
        # Generate queries with varying complexity
        research_queries = []
        
        # Track topic distribution
        topic_query_counts = {}
        
        # First pass: generate at least one query per topic to ensure diversity
        questions_by_topic = {}
        for q in questions:
            topic = q.get('topic', 'Unknown')
            if topic not in questions_by_topic:
                questions_by_topic[topic] = []
            questions_by_topic[topic].append(q)
        
        # Generate at least one query per topic
        for topic, topic_questions in questions_by_topic.items():
            if topic_questions:
                question = random.choice(topic_questions)
                complexity = random.randint(2, 3)  # Medium to high complexity
                query = self.generate_single_query(question, complexity)
                research_queries.append(query)
                
                topic_query_counts[topic] = topic_query_counts.get(topic, 0) + 1
        
        # Second pass: generate remaining queries
        remaining_count = target_count - len(research_queries)
        if remaining_count > 0:
            # Shuffle questions to ensure randomness
            random.shuffle(questions)
            
            # Generate multiple queries per question if needed
            for question in questions:
                for _ in range(queries_per_question):
                    if len(research_queries) >= target_count:
                        break
                        
                    complexity = random.randint(2, 3)  # Medium to high complexity
                    query = self.generate_single_query(question, complexity)
                    research_queries.append(query)
                    
                    topic = question.get('topic', 'Unknown')
                    topic_query_counts[topic] = topic_query_counts.get(topic, 0) + 1
                
                if len(research_queries) >= target_count:
                    break
        
        logger.info(f"Generated {len(research_queries)} research queries across {len(topic_query_counts)} topics")
        self.stats["final_count"] = len(research_queries)
        self.stats["final_topic_distribution"] = topic_query_counts
        return research_queries
    
    def generate_single_query(self, question, complexity=3):
        """Generate a single research query based on a filtered question"""
        question_text = question["text"]
        topic = question.get("topic", self.extract_topic_from_question(question_text))
        
        # Define task types
        TASK_TYPES = [
            "请分析{topic}的发展历史和未来趋势",
            "请查找关于{topic}的详细数据和分析",
            "请分析{topic}在不同国家或地区的差异",
            "请分析{topic}对社会/经济/文化的影响",
            "请比较{topic}的不同理论或方法",
            "请研究{topic}的最佳实践和案例",
            "请分析{topic}的法律法规和政策框架",
            "请分析{topic}的市场规模和竞争格局",
            "请分析{topic}的技术发展和创新方向",
            "请研究{topic}的风险因素和应对策略",
            "关于{topic}，需要全面的分析和研究",
            "针对{topic}，请提供深入的分析",
            "关于{topic}的问题，请进行详细研究",
            "请对{topic}进行多角度的分析"
        ]
        
        # Define specific requirements
        SPECIFIC_REQUIREMENTS = [
            "请按照时间顺序进行分析",
            "请按照地区进行分类比较",
            "请重点关注最近5年的变化",
            "请从多个角度进行分析",
            "请提供具体的数据支持",
            "请列出主要的优缺点",
            "请分析其中的因果关系",
            "请提供相关的案例研究",
            "请给出可行的解决方案",
            "请评估未来的发展趋势",
            "主要针对{topic}的核心领域",
            "不包括{topic}的边缘情况",
            "需要考虑不同人群的需求",
            "需要分析国内外的差异",
            "请考虑不同行业的应用场景",
            "请分析相关政策的影响"
        ]
        
        # Define output formats
        OUTPUT_FORMATS = [
            "请整理成一份报告",
            "请列成表格",
            "请做一个对比分析",
            "请总结主要观点",
            "请制作一个决策框架",
            "请提供一个分析框架",
            "请梳理成脉络清晰的内容",
            "请找出关键的影响因素",
            "请归纳成几个主要类别",
            "请提炼出核心的结论",
            "请按照具体的品类分类",
            "请提供详细的分析结果"
        ]
        
        # Optional persona prefixes (30% chance to include)
        PERSONA_PREFIXES = [
            "作为{role}，",
            "从{role}的角度，",
            "以{role}的身份，",
            ""  # Empty for no persona
        ]
        
        # Roles for personas
        ROLES = [
            "投资人", "研究者", "学生", "教育工作者", "医疗专业人士", 
            "科技行业从业者", "法律顾问", "环保工作者", "金融分析师", 
            "历史研究学者", "社会学研究者", "政策研究员", "文化产业从业者", 
            "旅游业从业者", "市场分析师", "创业者", "管理者"
        ]
        
        # Decide whether to include a persona (30% chance)
        has_persona = random.random() < 0.3
        
        if has_persona:
            persona_prefix = random.choice(PERSONA_PREFIXES[:-1])  # Exclude empty prefix
            role = random.choice(ROLES)
            query = persona_prefix.format(role=role)
        else:
            query = ""
        
        # Select a task type and format with the topic
        task = random.choice(TASK_TYPES).format(topic=topic)
        query += task
        
        # Add specific requirements based on complexity
        if complexity >= 2:
            num_requirements = random.randint(2, 4)
            requirements = random.sample(SPECIFIC_REQUIREMENTS, num_requirements)
            for req in requirements:
                req_formatted = req.format(topic=topic)
                query += "，" + req_formatted
        
        # Add output format request
        output_format = random.choice(OUTPUT_FORMATS)
        query += "，" + output_format
        
        return {
            "original_question": question_text,
            "research_query": query,
            "complexity": complexity,
            "has_persona": has_persona,
            "topic": topic
        }
    
    def extract_topic_from_question(self, question_text):
        """Extract the main topic from a question"""
        # Common topic indicators in Chinese questions
        indicators = ["如何", "为什么", "怎么", "是否", "有哪些", "如何看待", "怎样"]
        
        # Remove indicators to get the core topic
        topic = question_text
        for indicator in indicators:
            if indicator in topic:
                topic = topic.split(indicator, 1)[1]
        
        # Remove question marks and other punctuation
        topic = topic.replace("？", "").replace("?", "").strip()
        
        # Limit topic length
        if len(topic) > 15:
            topic = topic[:15]
        
        return topic
    
    def save_stage_output(self, data, filename):
        """Save the output of a stage to a JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {filepath}")

def main():
    """Run the Enhanced Chinese Research Query Pipeline"""
    # Initialize the pipeline
    pipeline = EnhancedQueryPipeline()
    
    # Run the pipeline with target of 500 queries
    research_queries = pipeline.run_pipeline(target_count=500)
    
    # Print sample queries
    print("\nSample of generated research queries:")
    for i, query in enumerate(random.sample(research_queries, min(5, len(research_queries)))):
        print(f"\n{i+1}. Topic: {query.get('topic', 'Unknown')}")
        print(f"   Original: {query['original_question']}")
        print(f"   Research Query: {query['research_query']}")
        print(f"   Complexity: {query['complexity']}")
        print(f"   Has Persona: {query['has_persona']}")
    
    # Print topic distribution
    topic_counts = {}
    for q in research_queries:
        topic = q.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("\nTopic distribution (top 20):")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{topic}: {count} queries")
    
    # Print complexity distribution
    complexity_counts = {}
    for q in research_queries:
        complexity = q.get('complexity', 'Unknown')
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    print("\nComplexity distribution:")
    for complexity, count in sorted(complexity_counts.items()):
        print(f"Complexity {complexity}: {count} queries ({count/len(research_queries)*100:.1f}%)")
    
    # Print persona distribution
    persona_count = sum(1 for q in research_queries if q.get('has_persona', False))
    no_persona_count = len(research_queries) - persona_count
    
    print("\nPersona distribution:")
    print(f"With persona: {persona_count} queries ({persona_count/len(research_queries)*100:.1f}%)")
    print(f"Without persona: {no_persona_count} queries ({no_persona_count/len(research_queries)*100:.1f}%)")

if __name__ == "__main__":
    main()
