from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=5):
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks

    def retrieve_multiple(self, queries, top_k=5):
        """Retrieve with multiple queries and deduplicate"""
        seen = set()
        unique_chunks = []

        for q in queries:
            for chunk in self.retrieve(q, top_k):
                if chunk['page_content'] not in seen:
                    seen.add(chunk['page_content'])
                    unique_chunks.append(chunk)

        return unique_chunks[:top_k]
    
def create_retriever(chunks, language):
    """Creates a BM25 retriever from document chunks."""
    return BM25Retriever(chunks, language)