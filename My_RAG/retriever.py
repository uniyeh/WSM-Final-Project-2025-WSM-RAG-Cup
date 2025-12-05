from rank_bm25 import BM25Okapi
import jieba
from nltk.stem import PorterStemmer

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

    def retrieve(self, query, top_k=5):
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokens = query.lower().split()
            tokenized_query = [self.stemmer.stem(token) for token in tokens]

        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks

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
    return chunks