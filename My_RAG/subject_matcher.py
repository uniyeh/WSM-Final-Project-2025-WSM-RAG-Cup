import os
import sys
from ollama import Client
from pathlib import Path
import yaml
import json
from typing import List, Dict, Tuple, Optional

# Import shared cache from router_utils
from router_utils import _DOCUMENT_CACHE, cache_document_names
from utils import load_ollama_config

def find_doc_names(query_text: str, language: str = "en", top_k: int = 3) -> Tuple[Optional[str], List[int]]:
    """
    Use LLM to match the query to the most relevant document names from the database.
    
    Args:
        query_text: The user's query
        language: Language filter ('en' or 'zh')
        top_k: Number of top matching documents to return
        
    Returns:
        Tuple of (domain, list of doc_ids)
    """
    global _DOCUMENT_CACHE
    
    if not _DOCUMENT_CACHE:
        _DOCUMENT_CACHE = cache_document_names(language)
        if (not _DOCUMENT_CACHE):
            print("No documents found in cache")
            return None
    
    # Prepare the list of document names
    doc_names = list(_DOCUMENT_CACHE.keys())
    
    # Create language-specific prompt for LLM
    if language == "zh":
        prompt = f"""你是一个文档匹配助手。给定用户查询和文档主题列表，识别哪些主题与回答查询高度相关。

用户查询：{query_text}

可用文档：
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(doc_names)])}

指示：
1. 分析查询以了解请求的信息
2. 将查询与列表中最相关的文档主题匹配
3. 仅返回主题编号（例如："1,5,12"），用逗号分隔
4. 如果没有明确相关的文档，返回"NONE"
5. 最多返回 {top_k} 个主题编号

你的答案（仅数字）："""
    else:  # English
        prompt = f"""You are a document matching assistant. Given a user query and a list of document subjects, identify which subjects are highly relevant to answer the query.

User Query: {query_text}

Available Documents:
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(doc_names)])}

Instructions:
1. Analyze the query to understand what information is being requested
2. Match the query to the most relevant document name(s) from the list
3. Return ONLY the subject numbers (e.g., "1,5,12") separated by commas
4. If no documents are clearly relevant, return "NONE"
5. Return at most {top_k} subject numbers

Your answer (numbers only):"""

    # Call LLM
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], prompt=prompt)
    
    # Parse response
    answer_idxes = response["response"].strip()
    print(f"LLM subject matching response: {answer_idxes}")
    
    if answer_idxes.upper() == "NONE" or not answer_idxes:
        print("No relevant documents found")
        return None
    
    # Parse indices with error handling
    try:
        match_doc_names = []
        for idx in answer_idxes.split(','):
            idx_int = int(idx.strip()) - 1
            if 0 <= idx_int < len(doc_names):
                match_doc_names.append(doc_names[idx_int])
            else:
                print(f"Warning: Index {idx_int + 1} out of range (max: {len(doc_names)})")
        
        if not match_doc_names:
            print("No valid document indices found")
            return None
            
        return match_doc_names
    except (ValueError, IndexError) as e:
        print(f"Error parsing LLM response indices: {e}")
        return None

if __name__ == "__main__":
    # Test the subject matcher
    print("Testing Subject Matcher...")
    
    # Cache documents
    cache = cache_document_names("en")
    print(f"\nCached {len(cache)} documents")
    print("\nSample documents:")
    for i, (name, info) in enumerate(list(cache.items())[:5]):
        print(f"  {name} -> {info['domain']}")
    
    # Test query
    test_query = "What are the regulations about financial reporting?"
    print(f"\nTest Query: {test_query}")
    
    doc_names = find_doc_names(test_query, language="en")
    print(f"\nResult:")
    print(f"  Document names: {doc_names}")
