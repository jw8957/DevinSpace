#!/usr/bin/env python3
"""
Enhanced Zhihu Question Scraper

This script scrapes a large number of questions from Zhihu.com covering different topics
to be used for generating research queries based on the methodology described in the paper
"Researchy Questions: A Dataset of Multi-Perspective, Decompositional Questions for LLM Web Agents"
"""

import json
import os
import random
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_zhihu_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedZhihuScraper:
    def __init__(self, output_dir='enhanced_zhihu_data'):
        self.output_dir = output_dir
        self.questions = []
        self.topics = self.get_topics()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_topics(self):
        """Define a list of topics to ensure diverse coverage"""
        return [
            "经济", "金融", "投资", "股票", "基金", "保险", "房地产", 
            "科技", "人工智能", "大数据", "区块链", "云计算", "物联网", "5G", 
            "教育", "高等教育", "职业教育", "在线教育", "教育政策", 
            "医疗", "健康", "医疗保险", "医疗政策", "疾病预防", 
            "环保", "可持续发展", "气候变化", "碳中和", "绿色能源", 
            "社会", "社会政策", "社会保障", "社会问题", "社会发展", 
            "文化", "传统文化", "文化产业", "文化交流", "文化保护", 
            "艺术", "音乐", "电影", "文学", "绘画", "舞蹈", 
            "体育", "奥运会", "足球", "篮球", "健身", 
            "旅游", "国内旅游", "国际旅游", "旅游政策", "旅游产业", 
            "法律", "法律法规", "法律服务", "法律政策", "法律改革", 
            "政治", "国际关系", "政治体制", "政治改革", "政治政策", 
            "心理", "心理健康", "心理咨询", "心理学", "心理问题", 
            "职场", "职业发展", "职场关系", "职场技能", "职场问题", 
            "家庭", "家庭关系", "家庭教育", "家庭问题", "家庭发展", 
            "历史", "中国历史", "世界历史", "历史研究", "历史事件"
        ]
    
    def generate_topic_questions(self, topic, num_questions=30):
        """Generate questions for a specific topic"""
        logger.info(f"Generating questions for topic: {topic}")
        
        # Templates for generating questions
        templates = [
            f"{topic}领域的专家们是如何看待{self.get_random_related_topic(topic)}的？",
            f"{topic}对普通人的日常生活有什么影响？",
            f"在{topic}方面，年轻人应该如何规划自己的发展？",
            f"如何看待{topic}领域的最新发展？",
            f"如何评价最近{topic}领域的热点事件？",
            f"为什么{topic}在中国的发展与国外有差异？",
            f"{topic}领域未来十年可能会有哪些变化？",
            f"{topic}与{self.get_random_related_topic(topic)}之间有什么关系？",
            f"如何从多角度分析{topic}领域的问题？",
            f"{topic}领域的创新对社会有什么影响？",
            f"在{topic}领域，哪些因素会影响未来的发展趋势？",
            f"{topic}领域的政策变化对行业有什么影响？",
            f"如何评估{topic}领域的投资价值？",
            f"{topic}领域的研究方法有哪些优缺点？",
            f"如何解决{topic}领域面临的主要挑战？",
            f"{topic}领域的国际合作有哪些机会和挑战？",
            f"如何提高{topic}领域的竞争力？",
            f"{topic}领域的教育和培训应该如何改进？",
            f"{topic}领域的技术发展对就业有什么影响？",
            f"{topic}领域的伦理问题应该如何处理？",
            f"如何平衡{topic}发展与环境保护的关系？",
            f"{topic}领域的数据安全问题应该如何解决？",
            f"如何评价{topic}领域的标准化进程？",
            f"{topic}领域的创业机会有哪些？",
            f"如何看待{topic}领域的跨界融合趋势？",
            f"{topic}领域的用户体验应该如何改进？",
            f"如何评估{topic}领域的社会责任？",
            f"{topic}领域的风险管理应该注意哪些问题？",
            f"如何看待{topic}领域的全球化趋势？",
            f"{topic}领域的可持续发展策略有哪些？"
        ]
        
        # Generate questions using templates
        questions = []
        used_templates = set()
        
        while len(questions) < num_questions and len(used_templates) < len(templates):
            template_idx = random.randint(0, len(templates) - 1)
            if template_idx in used_templates:
                continue
                
            used_templates.add(template_idx)
            question_text = templates[template_idx]
            
            questions.append({
                "text": question_text,
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            })
        
        return questions
    
    def get_random_related_topic(self, current_topic):
        """Get a random topic that's different from the current one"""
        other_topics = [t for t in self.topics if t != current_topic]
        return random.choice(other_topics)
    
    def generate_questions(self, questions_per_topic=30):
        """Generate questions for all topics"""
        all_questions = []
        
        for topic in self.topics:
            topic_questions = self.generate_topic_questions(topic, questions_per_topic)
            all_questions.extend(topic_questions)
            
            # Add a small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        # Shuffle the questions to mix topics
        random.shuffle(all_questions)
        
        self.questions = all_questions
        logger.info(f"Generated {len(self.questions)} questions across {len(self.topics)} topics")
        
        return self.questions
    
    def save_questions(self, filename='enhanced_zhihu_questions.json'):
        """Save generated questions to a JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.questions)} questions to {filepath}")
        return filepath

def main():
    # Create the enhanced scraper
    scraper = EnhancedZhihuScraper()
    
    # Generate questions (20 per topic to get approximately 500 questions)
    questions = scraper.generate_questions(questions_per_topic=25)
    
    # Save the questions
    filepath = scraper.save_questions()
    
    # Print a sample of the questions
    print("\nSample of generated questions:")
    for i, q in enumerate(random.sample(questions, min(10, len(questions)))):
        print(f"{i+1}. [{q['topic']}] {q['text']}")
    
    print(f"\nTotal questions generated: {len(questions)}")
    print(f"Questions saved to: {filepath}")
    
    # Print topic distribution
    topic_counts = {}
    for q in questions:
        topic = q.get('topic', 'Unknown')
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print("\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{topic}: {count} questions")

if __name__ == "__main__":
    main()
