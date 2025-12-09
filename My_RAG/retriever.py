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
        scores = scores.tolist()
        scores.sort(reverse=True)
        new_top_k = len(scores)
        if top1_check:
            for i in range(len(scores) - 1):
                if scores[0]/2 > scores[i+1]:
                    new_top_k = i+1
                    break
        else:
            for i in range(len(scores)):
                if scores[i] <= threshold:
                    new_top_k = i
                    break
        
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=new_top_k)
        # if top1_check and new_top_k > 1:
        #     print("scores: {}".format(scores))
        #     print("top_k: {}".format(new_top_k))
        #     print("threshold: {}".format(scores[new_top_k if new_top_k < len(scores) else len(scores) - 1]))
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