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
    prediction, doc_id, matched_name = name_router(query, language)
    if prediction:
        print("Router [1]:", prediction, doc_id, matched_name)
        return prediction, doc_id, matched_name

    # Second for vector router from query
    prediction_from_query_db, doc_id = embedding_query_db_router(query, language)
    print("Router [2]:", prediction_from_query_db, doc_id)

    # Specific router for pattern matching and content search
    prediction_from_query = embedding_query_router(query, language)
    prediction, total_doc_id = specific_router(query)
    if prediction and prediction==prediction_from_query_db and prediction==prediction_from_query:
        print("Router [3] with doc_id:", prediction)
        return prediction, doc_id, []
    elif prediction and prediction==prediction_from_query_db:
        print("Router [3]:", prediction)
        return prediction, total_doc_id, []
    elif prediction and prediction==prediction_from_query:
        print("Router [3]:", prediction)
        return prediction, total_doc_id, []
    elif prediction:
        print("Router [3]:", prediction)
        return prediction, total_doc_id, []

    print("Router [end]=====================")
    return None, [], []

def name_router(query, language="en"):
    content = query['query']['content']
    prediction = None
    doc_id = []
    matched_name = []

    # string matching logic
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
                matched_name.append(name)
        elif (name_docs[name]['domain'] == 'Medical'):
            if (name in content):
                prediction = 'Medical'
                doc_id.extend(name_docs[name]['doc_id'])
                matched_name.append(name)
        elif (name_docs[name]['domain'] == 'Finance'):
            if (name in content):
                prediction = 'Finance'
                doc_id.extend(name_docs[name]['doc_id']) 
                matched_name.append(name)
    if (prediction):
        return prediction, doc_id, matched_name

    # Try LLM-based matching
    """
    Use LLM to intelligently match query to document names.
    """
    from subject_matcher import find_doc_names
    from router_utils import cache_document_names    
    try:
        matched_names = find_doc_names(content, language=language, top_k=3)
        
        if matched_names:
            # Get the document cache to look up doc_ids and domain
            doc_cache = cache_document_names(language)
            for name in matched_names:
                if name in doc_cache:
                    doc_id.extend(doc_cache[name]['doc_ids'])
                    # Use the domain of the first matched document
                    if prediction is None:
                        prediction = doc_cache[name]['domain']
            
            if doc_id:
                print(f"✓ LLM name_router matched: {matched_names}")
                return prediction, doc_id, matched_names
    except Exception as e:
        print(f"LLM name_router failed: {e}, falling back to string matching")
    
    return None, [], []

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
        D, I = index.search(query_embedding, 1)

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
        prediction = max(domain_count, key=domain_count.get)
        if domain_count[prediction] < 3:
            prediction = None
        return prediction
    except Exception as e:
        print(f"Error in embedding_query_router: {e}")
        return None

def embedding_query_db_router(query, language="en"):
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
        cursor = conn.execute(f"SELECT domain, query_id, jsonl FROM queries WHERE id in ({placeholders})", id)
        rows = cursor.fetchall()
        doc_id = []
        domain_count = {
            "Law": 0,
            "Medical": 0,
            "Finance": 0
        }
        for row in rows:
            if row:
                jsonl = json.loads(row[2])
                doc_id.extend(jsonl['ground_truth']['doc_ids'])
                domain = row[0]
                domain_count[domain] += 1
        prediction = max(domain_count, key=domain_count.get)
        if domain_count[prediction] < 3:
            prediction = None
            doc_id = []
        return prediction, doc_id
    except Exception as e:
        print(f"Error in embedding_query_router: {e}")
        return None, []

# def bm25_router(query, language="en"):