from rank_bm25 import BM25Okapi
import jieba
from nltk.stem import PorterStemmer
import numpy as np

class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        
        self.stemmer = PorterStemmer()
        
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = []
            for doc in self.corpus:
                tokens = doc.lower().split()
                stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
                self.tokenized_corpus.append(stemmed_tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize_query(self, query):
        if self.language == "zh":
            return list(jieba.cut(query))
        else:
            tokens = query.lower().split()
            return [self.stemmer.stem(token) for token in tokens]

    def retrieve_dynamic_k(self, query, dominance_ratio=1.5, score_percentile=35):
        tokenized_query = self.tokenize_query(query)
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        scored_chunks = []
        for score, chunk in zip(doc_scores, self.chunks):
            # Print score for observation
            # print(f"Chunk Score: {score:.4f} | Content Snippet: '{chunk['page_content'][:30]}...'")
            if score > 0.0:
                chunk_with_score = chunk.copy()
                chunk_with_score['bm25_score'] = score
                scored_chunks.append(chunk_with_score)
        
        if not scored_chunks:
            return []
            
        scored_chunks.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        scores = np.array([chunk['bm25_score'] for chunk in scored_chunks])
        max_score = scores[0]
        
        if len(scores) > 1:
            second_max_score = scores[1]
            
            if max_score >= second_max_score * dominance_ratio:
                print(f"\n Detected dominant high score ({max_score:.4f}). Ratio to second max ({second_max_score:.4f}) is {max_score/second_max_score:.2f} (>{dominance_ratio}).")
                print("Returning only the highest scored chunk")
                return [scored_chunks[0]]
        
        threshold = np.percentile(scores, score_percentile)
        
        print(f"\nNo dominant score detected. Using the {score_percentile}th percentile of scores as threshold.")
        print(f"Dynamic Threshold ({score_percentile}th percentile): {threshold:.4f}")
        print("Returning chunks with score >= dynamic threshold")
        
        dynamic_top_chunks = [chunk for chunk in scored_chunks if chunk['bm25_score'] >= threshold]
        
        return dynamic_top_chunks

def create_retriever(chunks, language):
    """Creates a BM25 retriever from document chunks."""
    return BM25Retriever(chunks, language)

import sqlite3
import os
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../db/dataset.db'))

def get_chunks_from_db(prediction, doc_id, language):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if (doc_id):
        placeholders = ','.join('?' for _ in doc_id)
        cursor.execute(f"SELECT id, content FROM chunks WHERE doc_id IN ({placeholders})", doc_id)
        rows = cursor.fetchall()
        if not rows:
            return []
    elif (prediction):
        placeholders = ','.join('?' for _ in prediction)
        cursor.execute(f"SELECT id, content FROM chunks WHERE domain IN ({placeholders})", prediction)
        rows = cursor.fetchall()
        if not rows:
            return []
    else:
        cursor.execute("SELECT id, content FROM chunks where language = ?", (language,))
        rows = cursor.fetchall()
        if not rows:
            return []
    chunks = []
    for row in rows:
        chunks.append({"page_content": row[1]})
    
    conn.close()
    return chunks