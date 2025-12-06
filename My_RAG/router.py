import re
import numpy as np
from ollama import Client
import os
from utils import load_ollama_config
import faiss
import json
import sys
from router_utils import specific_router

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
DB_PATH = "db/dataset.db"

def router(query, language="en"):
    # First dispatch for query which contains name of document
    prediction, doc_id = name_router(query, language)
    if prediction:
        print("Router [1]:", prediction, doc_id)
        return prediction, doc_id

    # Second for vector router from query
    prediction_from_query = embedding_query_router(query, language)
    if (prediction_from_query):
        print("Router [2]:", prediction_from_query, [])

    # Specific router for pattern matching and content search
    prediction, doc_id = specific_router(query)
    if prediction and prediction_from_query==prediction:
        print("Router [3]:", prediction, doc_id)
        return prediction, doc_id
    print("Router [end]=====================")
    return None, []

def name_router(query, language="en"):
    """
    Use LLM to intelligently match query to document names.
    Falls back to simple string matching if LLM fails.
    """
    from subject_matcher import find_doc_names
    from router_utils import cache_document_names
    
    content = query['query']['content']
    
    # Try LLM-based matching first
    try:
        matched_names = find_doc_names(content, language=language, top_k=3)
        
        if matched_names:
            # Get the document cache to look up doc_ids and domain
            doc_cache = cache_document_names(language)
            
            prediction = None
            doc_id = []
            
            for name in matched_names:
                if name in doc_cache:
                    doc_id.extend(doc_cache[name]['doc_ids'])
                    # Use the domain of the first matched document
                    if prediction is None:
                        prediction = doc_cache[name]['domain']
            
            if doc_id:
                print(f"✓ LLM name_router matched: {matched_names}")
                return prediction, doc_id
    except Exception as e:
        print(f"LLM name_router failed: {e}, falling back to string matching")
    
    # Fallback: Original string matching logic
    conn = Connection(DB_PATH)
    rows = conn.execute("SELECT domain, name, doc_id FROM documents WHERE language = ?", (language,))
    name_docs = {}
    for row in rows:
        domain = row[0]
        name = row[1]
        if name not in name_docs:
            name_docs[name] = {
                'doc_id': [],
                'domain': domain
            }
        name_docs[name]['doc_id'].append(row[2])

    prediction = None
    doc_id = []
    for name in name_docs:
        if (name_docs[name]['domain'] == 'Law'):
            match = False
            if (language == 'en'):
                split_name = name.split(',')
                match = True
                for i in range(1, len(split_name)):
                    if split_name[i] not in content:
                        match = False
                        break
            else:
                match = name in content
            if (match):
                prediction = 'Law'
                doc_id.extend(name_docs[name]['doc_id'])
        elif (name_docs[name]['domain'] == 'Medical'):
            if (name in content):
                prediction = 'Medical'
                doc_id.extend(name_docs[name]['doc_id'])
        elif (name_docs[name]['domain'] == 'Finance'):
            if (name in content):
                prediction = 'Finance'
                doc_id.extend(name_docs[name]['doc_id'])         
    return prediction, doc_id

def get_embedding(text, language="en"):
    ollama_config = load_ollama_config()
    ollama_host = ollama_config["host"]
    client = Client(host=ollama_host)
    # Using the same model as generator/router_llm
    response = client.embeddings(model="qwen3-embedding:0.6b", prompt=text)
    return np.array(response['embedding'], dtype='float32').reshape(1, -1)

def embedding_query_router(query, language="en"):
    content = query['query']['content']
    try:
        query_embedding = get_embedding(content, language)

        # Load FAISS index
        index = faiss.read_index("db/faiss/queries/" + language + "/" + language + ".index")

        # Search for nearest neighbors 
        D, I = index.search(query_embedding, 5)

        # Get mapping from FAISS ID to document ID
        mapping_path = "db/faiss/queries/" + language + "/" + language + "_mapping.json"
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        id = [mapping[str(id)] for id in I[0]]

        # Get document
        conn = Connection(DB_PATH)
        placeholders = ','.join('?' for _ in id)
        # Query chunks table because FAISS index is built from chunks
        cursor = conn.execute(f"SELECT domain, query_id FROM queries WHERE id in ({placeholders})", id)
        rows = cursor.fetchall()
        domain_count = {
            "Law": 0,
            "Medical": 0,
            "Finance": 0
        }
        for row in rows:
            if row:
                domain = row[0]
                domain_count[domain] += 1
        print(domain_count)
        prediction = max(domain_count, key=domain_count.get)
        if domain_count[prediction] < 3:
            prediction = None
        return prediction
    except Exception as e:
        print(f"Error in embedding_query_router: {e}")
        return None, []
