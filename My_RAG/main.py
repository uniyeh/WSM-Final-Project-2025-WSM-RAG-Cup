from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from generator import generate_answer
import argparse
from router import router
from embedding_retriever import embedding_retriever
from router_utils import cache_document_names
from rank_bm25 import BM25Okapi

def main(query_path, docs_path, language, output_path):
    # 0. Cache document names at startup (for LLM-based routing)
    print("Caching document names from database...")
    cache_document_names(language)
    
    # 1. Load Queries
    queries = load_jsonl(query_path)

    for query in tqdm(queries, desc="Processing Queries"):
        # 1. Route query
        print("Routing query[{}]...".format(query['query']['query_id']))
        query_text = query['query']['content']
        prediction, doc_id, doc_names = router(query, language)

        # 2. Retrieve relevant chunks(use embedding retriever)
        print("Retrieving chunks...")
        # if (doc_id):
        #     retrieved_chunks = embedding_retriever(query_text, language, domain=prediction, doc_ids=doc_id)
        # elif (prediction):
        #     retrieved_chunks = embedding_retriever(query_text, language, domain=prediction)
        # else:
        #     retrieved_chunks = embedding_retriever(query_text, language, top_k=5)
        
        # 2. Retrieve relevant chunks(use BM25)
        row_chunks = get_chunks_from_db(prediction, doc_id, language)
        retriever = create_retriever(row_chunks, language)
        
        print("[1]first retrieve for query: {}".format(query_text))
        retrieved_chunks = retriever.retrieve(query_text, top_k=1)

        small_chunks = chunk_row_chunks(retrieved_chunks, language)
        if (doc_names):
            for doc_name in doc_names:
                query_text = query_text.replace(doc_name, "")
        small_retrieved_chunks = []
        for index, chunk in enumerate(small_chunks):
            if (doc_names):
                for doc_name in doc_names:
                    small_retrieved_chunks.append({
                        "page_content": chunk['page_content'].replace(doc_name, ""),
                        "chunk_index": index
                    })
            else:
                small_retrieved_chunks.append({
                    "page_content": chunk['page_content'],
                    "chunk_index": index
                })

        print("[2]second retrieve for query: {}".format(query_text))
        retriever_2 = create_retriever(small_retrieved_chunks, language)
        retrieved_small_chunks = retriever_2.retrieve(query_text, top1_check=True)
        return_chunks = []
        for index, chunk in enumerate(retrieved_small_chunks):
            return_chunks.append(small_chunks[chunk['chunk_index']])
            
        # 3. Generate Answer
        print("Generating answer...")
        print('retrieved_chunks', return_chunks)
        answer = generate_answer(query['query']['content'], return_chunks, language)
        if ("无法回答" not in answer and 'Unable to answer' not in answer):
            retrieve_answer = answer
            if (doc_names):
                for doc_name in doc_names:
                    retrieve_answer = answer.replace(doc_name, "")
                final_retrieve = query_text + " " + retrieve_answer
            else:
                final_retrieve = query['query']['content'] + " " + answer
            print("[3]third retrieve for answer: {}".format(final_retrieve))
            retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True)
            return_chunks = []
            for index, chunk in enumerate(retrieved_small_chunks):
                return_chunks.append(small_chunks[chunk['chunk_index']])
        
        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [chunk["page_content"] for chunk in return_chunks]

    save_jsonl(output_path, queries)
    print("Predictions saved.")
    print(output_path)
    print("=====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
