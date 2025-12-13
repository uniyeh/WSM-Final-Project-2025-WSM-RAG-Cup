import re
from typing import Dict, List, Optional
from ollama import Client
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from My_RAG.utils import load_ollama_config

def extract_entities_with_llm(query: str, language: str = "en") -> Dict[str, List[str]]:
    """
    Extract entities from query using LLM.
    
    Args:
        query: User query text
        language: Language code ('en' or 'zh')
        
    Returns:
        Dictionary with extracted entities:
        {
            'years': ['2019', '2020'],
            'months': ['March', 'April'],
            'dates': ['2019-03-15', '2020-04-20'],
            'people': ['James Peterson', 'Sarah Chen'],
            'companies': ['National Development Corporation']
        }
    """
    if language == "zh":
        prompt = f"""从以下查询中提取所有实体。以JSON格式返回结果。

查询：{query}

请提取：
1. 年份（例如：2019, 2020）
2. 月份（例如：3月, 4月, 三月）
3. 完整日期（例如：2019年3月15日）
4. 人名（例如：张三, 李四）
5. 公司名称（例如：绿源环保有限公司）

输出格式（JSON）：
{{
    "years": ["2019", "2020"],
    "months": ["3月", "4月"],
    "dates": ["2019年3月15日"],
    "people": ["张三"],
    "companies": ["绿源环保有限公司"]
}}

如果某类实体不存在，返回空列表[]。
只返回JSON，不要其他文字。"""
    else:
        prompt = f"""Extract all entities from the following query. Return the result in JSON format.

Query: {query}

Extract:
1. Years (e.g., 2019, 2020)
2. Months (e.g., March, April)
3. Full dates (e.g., March 15, 2019)
4. People names (e.g., James Peterson, Sarah Chen)
5. Company names (e.g., National Development Corporation)

Output format (JSON):
{{
    "years": ["2019", "2020"],
    "months": ["March", "April"],
    "dates": ["March 15, 2019"],
    "people": ["James Peterson"],
    "companies": ["National Development Corporation"]
}}

If a category has no entities, return an empty list [].
Return ONLY JSON, no other text."""

    try:
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        response = client.generate(model=ollama_config["model"], prompt=prompt)
        
        # Parse JSON response
        import json
        response_text = response["response"].strip()
        
        # Try to extract JSON from response
        # Sometimes LLM adds extra text, so find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group())
        else:
            entities = json.loads(response_text)
        
        # Ensure all keys exist
        default_entities = {
            'years': [],
            'months': [],
            'dates': [],
            'people': [],
            'companies': []
        }
        default_entities.update(entities)
        
        return default_entities
        
    except Exception as e:
        print(f"Error extracting entities with LLM: {e}")
        return {
            'years': [],
            'months': [],
            'dates': [],
            'people': [],
            'companies': []
        }


def extract_entities_with_regex(query: str, language: str = "en") -> Dict[str, List[str]]:
    """
    Extract entities using regex patterns (faster but less accurate).
    
    Args:
        query: User query text
        language: Language code ('en' or 'zh')
        
    Returns:
        Dictionary with extracted entities
    """
    entities = {
        'years': [],
        'months': [],
        'dates': [],
        'people': [],
        'companies': []
    }
    
    if language == "zh":
        # Extract years (4-digit numbers that look like years)
        years = re.findall(r'(19\d{2}|20\d{2})年?', query)
        entities['years'] = list(set(years))
        
        # Extract months
        months = re.findall(r'([1-9]|1[0-2])月', query)
        entities['months'] = [f"{m}月" for m in set(months)]
        
        # Extract full dates (e.g., 2019年3月15日)
        dates = re.findall(r'((?:19|20)\d{2})年([1-9]|1[0-2])月([1-9]|[12]\d|3[01])日', query)
        entities['dates'] = [f"{y}年{m}月{d}日" for y, m, d in dates]
        
        # Extract potential company names (ending with 公司, 有限公司, etc.)
        companies = re.findall(r'[\u4e00-\u9fa5]{2,}(?:有限公司|股份有限公司|公司|集团)', query)
        entities['companies'] = list(set(companies))
        
    else:  # English
        # Extract years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        entities['years'] = list(set(years))
        
        # Extract months
        month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        months = re.findall(month_pattern, query, re.IGNORECASE)
        entities['months'] = list(set(months))
        
        # Extract full dates (various formats)
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, query, re.IGNORECASE)
            if isinstance(dates[0] if dates else None, tuple):
                entities['dates'].extend([' '.join(d) if isinstance(d, tuple) else d for d in dates])
            else:
                entities['dates'].extend(dates)
        
        # Extract capitalized names (potential people/companies)
        # This is a simple heuristic - capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)
        
        # Try to distinguish between people and companies
        company_keywords = ['Corporation', 'Ltd', 'Limited', 'Inc', 'Company', 'Industries', 'Group']
        for name in capitalized:
            if any(keyword in name for keyword in company_keywords):
                entities['companies'].append(name)
            else:
                entities['people'].append(name)
    
    return entities


def extract_entities(query: str, language: str = "en", use_llm: bool = True) -> Dict[str, List[str]]:
    """
    Extract entities from query.
    
    Args:
        query: User query text
        language: Language code ('en' or 'zh')
        use_llm: If True, use LLM extraction; if False, use regex
        
    Returns:
        Dictionary with extracted entities
    """
    if use_llm:
        return extract_entities_with_llm(query, language)
    else:
        return extract_entities_with_regex(query, language)


if __name__ == "__main__":
    # Test entity extraction
    print("=== Testing Entity Extraction ===\n")
    
    # Test Chinese
    zh_query = "绿源环保有限公司在2017年4月修订了什么政策？"
    print(f"Chinese Query: {zh_query}")
    print("Regex extraction:")
    print(extract_entities(zh_query, "zh", use_llm=False))
    print("\nLLM extraction:")
    print(extract_entities(zh_query, "zh", use_llm=True))
    
    print("\n" + "="*50 + "\n")
    
    # Test English
    en_query = "When did James Peterson join National Development Corporation in March 2019?"
    print(f"English Query: {en_query}")
    print("Regex extraction:")
    print(extract_entities(en_query, "en", use_llm=False))
    print("\nLLM extraction:")
    print(extract_entities(en_query, "en", use_llm=True))
