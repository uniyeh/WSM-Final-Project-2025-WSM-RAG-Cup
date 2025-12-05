import os
import sys
import json
import numpy as np
import faiss
import sqlite3
from ollama import Client
from utils import load_ollama_config

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../db/dataset.db'))

def get_query_embedding(text):
    ollama_config = load_ollama_config()
    ollama_host = ollama_config["host"]
    client = Client(host=ollama_host)
    # Using the same model as generator/router_llm
    response = client.embeddings(model="qwen3-embedding:0.6b", prompt=text)
    return np.array(response['embedding'], dtype='float32').reshape(1, -1)

def get_chunks_rows(language="en", doc_ids=None, domain=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if domain and doc_ids:
        placeholders = ','.join('?' for _ in doc_ids)
        cursor.execute(f"SELECT id, content FROM chunks WHERE doc_id IN ({placeholders})", doc_ids)
        rows = cursor.fetchall()
        if not rows:
            return []
    elif domain:
        placeholders = ','.join('?' for _ in domain)
        cursor.execute(f"SELECT id, content FROM chunks WHERE domain IN ({placeholders})", domain)
        rows = cursor.fetchall()
        if not rows:
            return []
    else:
        cursor.execute("SELECT id, content FROM chunks")
        rows = cursor.fetchall()
        if not rows:
            return []
    return rows

def embedding_retriever(query, language="en", doc_ids=None, domain=None, top_k=5, threshold=0.8):
    """
    Retrieve chunks using embedding similarity.
    If doc_ids is provided, filter chunks by these chunks first.
    threshold: float, optional. If provided, only return chunks with L2 distance <= threshold.
               Note: Since we use L2 distance, lower is better. 'Score too low' corresponds to 'Distance too high'.
    """
    # Get query embedding
    query_embedding = get_query_embedding(query)
    
    # Paths
    base_dir = os.path.dirname(__file__)
    index_path = os.path.join(base_dir, f"../db/faiss/chunks/{language}/{language}.index")
    mapping_path = os.path.join(base_dir, f"../db/faiss/chunks/{language}/{language}_mapping.json")
    
    # Load index and mapping
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        print(f"Error: FAISS index or mapping not found at {index_path}")
        return []

    index = faiss.read_index(index_path)
    with open(mapping_path, 'r') as f:
        mapping = json.load(f) # FAISS ID (str) -> Chunk ID (int)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    rows = get_chunks_rows(language, doc_ids, domain)
    if not rows:
        return []

    chunk_ids = [row[0] for row in rows]
    chunk_contents = {row[0]: row[1] for row in rows}
    
    # 2. Invert mapping to find FAISS IDs for these chunk IDs
    # mapping is str(faiss_id) -> chunk_id
    # We need chunk_id -> faiss_id
    chunk_to_faiss = {v: int(k) for k, v in mapping.items()}
    
    vectors = []
    valid_chunk_ids = []
    
    for cid in chunk_ids:
        if cid in chunk_to_faiss:
            fid = chunk_to_faiss[cid]
            try:
                # Reconstruct vector from FAISS index
                vec = index.reconstruct(fid)
                vectors.append(vec)
                valid_chunk_ids.append(cid)
            except Exception as e:
                print(f"Error reconstructing vector for chunk {cid} (FAISS ID {fid}): {e}")
    
    if not vectors:
        return []
        
    vectors_np = np.array(vectors)
    
    # 3. Calculate similarity (L2 distance)
    # query_embedding is (1, d), vectors_np is (n, d)
    # L2 = sum((q - v)^2)
    diff = vectors_np - query_embedding
    dists = np.sum(diff**2, axis=1)
    
    # Filter by threshold if provided
    if threshold is not None:
        # Keep indices where dist <= threshold
        valid_indices = np.where(dists <= threshold)[0]
        dists = dists[valid_indices]
        # Update vectors_np or just map back to chunk_ids using valid_indices
        # We need to filter valid_chunk_ids as well
        valid_chunk_ids = [valid_chunk_ids[i] for i in valid_indices]
        
        if not valid_chunk_ids:
            return []

    # 4. Sort and get top_k
    sorted_indices = np.argsort(dists)
    # If top_k is larger than available chunks, take all
    k = min(top_k, len(sorted_indices))
    top_indices = sorted_indices[:k]
    
    results = []
    for idx in top_indices:
        cid = valid_chunk_ids[idx]
        results.append(chunk_contents[cid])
    
    return results
