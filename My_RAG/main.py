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
        print("Routing query[{}]: {}".format(query['query']['query_id'], query['query']['content']))
        # Route query to chains
        answer, return_chunks = router(query, language)
        # save answer and chunks
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
