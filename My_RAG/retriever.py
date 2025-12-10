from rank_bm25 import BM25Okapi
import jieba
from nltk.stem import PorterStemmer
import re

class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        self.stopwords = []
        self.stemmer = PorterStemmer()
        
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stopwords_path = os.path.join(script_dir, 'english.stop')
            self.stopwords = open(stopwords_path, 'r').read().split()
            self.tokenized_corpus = []
            for doc in self.corpus:
                tokens = self.clean(doc).split()
                tokens = [token for token in tokens if token not in self.stopwords]
                stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
                self.tokenized_corpus.append(stemmed_tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def clean(self, string):
        string = string.lower()
        # Remove punctuation
        string = re.sub(r"[.,!?;:'\"()]", " ", string)
        # Normalize whitespace
        string = re.sub(r"\s+", " ", string)
        return string.strip()

    def retrieve(self, query, top_k=5, top1_check=False, threshold=0):
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokens = self.clean(query).split()
            tokens = [token for token in tokens if token not in self.stopwords]
            tokenized_query = [self.stemmer.stem(token) for token in tokens]

        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Debug: Show top scores
        top_scores = [scores[i] for i in top_indices]
        print(f"[BM25] Top {len(top_scores)} scores: {[f'{s:.2f}' for s in top_scores[:5]]}")
        
        # Filter by threshold (since sorted, can cut off when score drops below threshold)
        if threshold > 0:
            filtered_indices = []
            for idx in top_indices:
                if scores[idx] > threshold:
                    filtered_indices.append(idx)
                else:
                    break  # Scores are sorted, so we can stop here
            top_indices = filtered_indices
        
        # Apply top1_check if needed
        if top1_check and len(top_indices) > 1:
            top_score = scores[top_indices[0]]
            # Keep only chunks with score > top_score/2
            filtered = []
            for idx in top_indices:
                if scores[idx] > top_score/2:
                    filtered.append(idx)
                else:
                    break
            top_indices = filtered
        
        # Get the actual chunks
        top_chunks = [self.chunks[i] for i in top_indices]
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
    if (prediction and doc_id):
        placeholders = ','.join('?' for _ in doc_id)
        cursor.execute(f"SELECT id, name, content FROM chunks WHERE doc_id IN ({placeholders})", doc_id)
        rows = cursor.fetchall()
        if not rows:
            return []
    elif (prediction):
        cursor.execute(f"SELECT id, name, content FROM chunks WHERE domain = ? and language = ?", (prediction, language))
        rows = cursor.fetchall()
        if not rows:
            return []
    else:
        cursor.execute("SELECT id, name, content FROM chunks where language = ?", (language,))
        rows = cursor.fetchall()
        if not rows:
            return []
    chunks = []
    rows = [list(row) for row in rows]
    for index, row in enumerate(rows):
        if (index == len(rows) - 1):
            chunks.append({"page_content": row[2], "name": row[1]})
            break
        if language == "zh":
            if (len(row[2]) < 10 and index < len(rows) - 1):
                # together with the next chunk
                rows[index+1][2] = row[2] +'. ' + rows[index+1][2]
                continue
        else:
            if (len(row[2]) < 30 and index < len(rows) - 1):
                # together with the next chunk
                rows[index+1][2] = row[2] +'. ' + rows[index+1][2]
                continue
        chunks.append({"page_content": row[2], "name": row[1]})
    return chunks