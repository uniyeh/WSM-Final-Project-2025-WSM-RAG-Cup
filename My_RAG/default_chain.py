import re
import numpy as np
from ollama import Client
import os
import sys
from utils import load_ollama_config
import faiss
import json
from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from generator import generate_answer
from rank_bm25 import BM25Okapi
from router_utils import specific_router

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
DB_PATH = "db/dataset.db"

def default_chain(query, language="en", prediction=None, doc_id=[], doc_names=[]):
    prediction_from_query_db, doc_id = embedding_query_db_router(query, language)
    print("prediction_from_query_db: ", prediction_from_query_db, "doc_id: ", doc_id)
    prediction_from_query = embedding_query_router(query, language)
    print("prediction_from_query: ", prediction_from_query)
    prediction, total_doc_id = specific_router(query)
    print("prediction: ", prediction, "total_doc_id: ", total_doc_id)

    if prediction!=prediction_from_query_db and prediction!=prediction_from_query:
        doc_id = total_doc_id

    query_text = query['query']['content']
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    print("query_text: ", query_text)
    print("modified_query_text: ", modified_query_text)

    # 1. Retrieve bigger chunks(use BM25)
    row_chunks = get_chunks_from_db(prediction, doc_id, language)
    retriever = create_retriever(row_chunks, language)
    
    print("[1] retrieve with bigger chunks:")
    retrieved_chunks = retriever.retrieve(query_text, threshold=0) # retrieve as much as possible
    print('chunks: ', len(retrieved_chunks))

    # 2. Retrieve smaller chunks(use BM25)
    print("[2] retrieve with smaller chunks and extract document name:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])

    print('chunks: ', len(return_chunks))

    # 3. Generate Answer
    print("[3] generate answer:")
    answer = generate_answer(query['query']['content'], return_chunks, language)
    if ("无法回答" in answer or 'Unable to answer' in answer):
        return answer, return_chunks
    
    #4. Fine-tune retriever
    retrieve_answer = get_remove_names_from_text(answer, doc_names)
    final_retrieve = modified_query_text + " " + retrieve_answer
    print("[4] rerieve for final answer: {}".format(final_retrieve))
    retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    print('final chunks: ', len(return_chunks))
    return answer, return_chunks

########## Helper Functions ##########

def create_smaller_chunks_without_names(language="en", retrieved_chunks=[], doc_names=[]):
    small_chunks = chunk_row_chunks(retrieved_chunks, language)
    small_retrieved_chunks = []
    for index, chunk in enumerate(small_chunks):
        small_retrieved_chunks.append({
            "page_content": get_remove_names_from_text(chunk['page_content'], doc_names),
            "chunk_index": index
        })
    return small_retrieved_chunks, small_chunks

def get_remove_names_from_text(content, doc_names = []):
    if (doc_names):
        for doc_name in doc_names:
            content = content.replace(doc_name, "")
    return content


def get_embedding(text, language="en"):
    ollama_config = load_ollama_config()
    ollama_host = ollama_config["host"]
    client = Client(host=ollama_host)
    # Using the same model as generator/router_llm
    response = client.embeddings(model="qwen3-embedding:0.6b", prompt=text)
    return np.array(response['embedding'], dtype='float32').reshape(1, -1)

def embedding_query_router(query, language="en"):
    content = query['query']['content']
    prediction = None
    try:
        query_embedding = get_embedding(content, language)

        # Load FAISS index
        index = faiss.read_index("db/faiss/documents/" + language + "/" + language + ".index")

        # Search for nearest neighbors 
        D, I = index.search(query_embedding, 1)

        # Get mapping from FAISS ID to document ID
        mapping_path = "db/faiss/documents/" + language + "/" + language + "_mapping.json"
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        id = [mapping[str(id)] for id in I[0]]

        # Get document
        conn = Connection(DB_PATH)
        placeholders = ','.join('?' for _ in id)
        # Query chunks table because FAISS index is built from chunks
        cursor = conn.execute(f"SELECT domain FROM documents WHERE id in ({placeholders})", id)
        row = cursor.fetchone()
        if row:
            prediction = row[0]
    except Exception as e:
        print(f"Error in embedding_query_router: {e}")
    return prediction

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