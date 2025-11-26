from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from multi_query import generate_multi_queries
import argparse

def main(query_path, docs_path, language, output_path, use_multi_query=False):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']

        if use_multi_query:
            print(f"\n{'='*80}")
            print(f"Original Query: {query_text}")
            print(f"{'-'*80}")
            query_variations = generate_multi_queries(query_text, num_queries=3, language=language)
            print(f"Generated {len(query_variations)} query variations:")
            for i, q in enumerate(query_variations):
                print(f"  {i+1}. {q}")
            print(f"{'-'*80}")
            retrieved_chunks = retriever.retrieve_multiple(query_variations)
            print(f"Retrieved {len(retrieved_chunks)} unique chunks using multi-query")
            print(f"{'='*80}\n")
        else:
            retrieved_chunks = retriever.retrieve(query_text)
            print(f"Retrieved {len(retrieved_chunks)} chunks using single query")

        # 5. Generate Answer
        # print("Generating answer...")
        answer = generate_answer(query_text, retrieved_chunks)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [retrieved_chunks[0]['page_content']]

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')      
    parser.add_argument('--use_multi_query', action='store_true', help='Enable multi-query')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output, args.use_multi_query)
