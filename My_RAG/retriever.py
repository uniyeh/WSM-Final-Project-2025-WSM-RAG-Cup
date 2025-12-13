from rank_bm25 import BM25Okapi
import jieba
from nltk.stem import PorterStemmer
import re
from ollama import Client
from utils import load_ollama_config

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
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        # Filter by threshold (since sorted, can cut off when score drops below threshold)
        if threshold > -1:
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
            chunks.append({"id": row[0], "page_content": row[2], "name": row[1]})
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
        chunks.append({"id": row[0], "page_content": row[2], "name": row[1]})
    return chunks

class DenseRetriever:
    """Dense retriever using FAISS pre-computed embeddings."""
    
    def __init__(self, chunks, language="en", embedding_model="qwen3-embedding:0.6b", use_faiss=True):
        """
        Initialize dense retriever.
        
        Args:
            chunks: List of document chunks
            language: Language code ('en' or 'zh')
            embedding_model: Embedding model name (must match FAISS: qwen3-embedding:0.6b)
            use_faiss: If True, load FAISS index; if False, generate embeddings on-the-fly
        """
        self.chunks = chunks
        self.language = language
        self.embedding_model = embedding_model
        self.use_faiss = use_faiss
        self.corpus = [chunk['page_content'] for chunk in chunks]
        
        # Load Ollama configuration for host
        ollama_config = load_ollama_config()
        self.client = Client(host=ollama_config["host"])
        
        if use_faiss:
            # Load pre-computed FAISS index
            import faiss
            import json
            from pathlib import Path
            
            faiss_dir = Path(__file__).parent.parent / "db" / "faiss" / "chunks" / language
            index_path = faiss_dir / f"{language}.index"
            mapping_path = faiss_dir / f"{language}_mapping.json"
            
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
            print(f"[DenseRetriever] Loading FAISS index from {index_path}...")
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load mapping (FAISS ID -> Chunk ID)
            with open(mapping_path, 'r') as f:
                self.faiss_to_chunk_id = json.load(f)
                # Convert string keys to int
                self.faiss_to_chunk_id = {int(k): v for k, v in self.faiss_to_chunk_id.items()}
            
            # Create chunk_id to index mapping
            self.chunk_id_to_idx = {chunk['id'] if 'id' in chunk else i: i 
                                   for i, chunk in enumerate(chunks)}
            
            print(f"[DenseRetriever] Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Generate embeddings on-the-fly (slow!)
            print(f"[DenseRetriever] Generating embeddings for {len(self.corpus)} chunks using {embedding_model}...")
            self.chunk_embeddings = []
            for doc in self.corpus:
                response = self.client.embeddings(model=self.embedding_model, prompt=doc)
                self.chunk_embeddings.append(response['embedding'])
            print(f"[DenseRetriever] Embeddings generated successfully")

    def cosine_similarity(self, vec1, vec2):
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def retrieve(self, query, top_k=5, top1_check=False, threshold=0):
        """
        Retrieve top-k most similar chunks using embedding similarity.
        Caches the similarity results for later use with get_scores().
        
        Args:
            query: Search query string
            top_k: Number of top chunks to return
            top1_check: If True, filter by top score ratio
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of top-k most similar chunks
        """
        # Generate query embedding
        query_response = self.client.embeddings(model=self.embedding_model, prompt=query)
        query_embedding = query_response['embedding']
        
        if self.use_faiss:
            # Use FAISS search
            import numpy as np
            query_vec = np.array([query_embedding], dtype='float32')
            
            # Search FAISS index (returns L2 distances, not cosine similarity)
            distances, faiss_indices = self.faiss_index.search(query_vec, min(top_k * 2, self.faiss_index.ntotal))
            
            # Convert L2 distances to cosine similarities (approximate)
            # For normalized vectors: cosine_sim ≈ 1 - (L2_dist^2 / 2)
            similarities = []
            for faiss_idx, dist in zip(faiss_indices[0], distances[0]):
                if faiss_idx == -1:  # FAISS returns -1 for empty results
                    continue
                chunk_id = self.faiss_to_chunk_id.get(int(faiss_idx))
                if chunk_id is not None and chunk_id in self.chunk_id_to_idx:
                    chunk_idx = self.chunk_id_to_idx[chunk_id]
                    # Convert L2 distance to approximate cosine similarity
                    sim = 1 - (dist / 2)
                    similarities.append((chunk_idx, sim))
        else:
            # Calculate similarities on-the-fly
            similarities = []
            for i, chunk_embedding in enumerate(self.chunk_embeddings):
                sim = self.cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((i, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the last query and similarities for get_scores()
        self._last_query = query
        self._last_similarities = similarities
        
        # Get top_k indices
        top_indices = [idx for idx, sim in similarities[:top_k]]
        top_scores = [sim for idx, sim in similarities[:top_k]]
        
        print(f"[DenseRetriever] Top {len(top_scores)} similarities: {[f'{s:.3f}' for s in top_scores[:5]]}")
        
        # Filter by threshold
        if threshold > 0:
            filtered_indices = []
            filtered_scores = []
            for idx, score in zip(top_indices, top_scores):
                if score > threshold:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)
                else:
                    break
            print(f"[DenseRetriever] Threshold={threshold}: {len(top_indices)} → {len(filtered_indices)} chunks")
            top_indices = filtered_indices
            top_scores = filtered_scores
        
        # Apply top1_check if needed
        if top1_check and len(top_indices) > 1 and len(top_scores) > 0:
            top_score = top_scores[0]
            filtered_indices = []
            filtered_scores = []
            for idx, score in zip(top_indices, top_scores):
                if score > top_score / 2:
                    filtered_indices.append(idx)
                    filtered_scores.append(score)
                else:
                    break
            top_indices = filtered_indices
            top_scores = filtered_scores
        
        # Cache the final results
        self._last_top_indices = top_indices
        self._last_top_scores = top_scores
        
        # Get the actual chunks
        top_chunks = [self.chunks[i] for i in top_indices]
        return top_chunks
    
    def get_scores(self):
        """
        Get the similarity scores from the last retrieve() call.
        Must be called after retrieve().
        
        Returns:
            List of similarity scores for the retrieved chunks
        """
        if not hasattr(self, '_last_top_scores'):
            raise ValueError("No retrieval has been performed yet. Call retrieve() first.")
        return self._last_top_scores
    
    def get_all_scores(self):
        """
        Get all similarity scores from the last retrieve() call (not just top-k).
        Must be called after retrieve().
        
        Returns:
            List of (index, score) tuples sorted by score (descending)
        """
        if not hasattr(self, '_last_similarities'):
            raise ValueError("No retrieval has been performed yet. Call retrieve() first.")
        return self._last_similarities