from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
import argparse
from router import router
from embedding_retriever import embedding_retriever

def main(query_path, docs_path, language, output_path):
    # 1. Load Queries
    queries = load_jsonl(query_path)

    for query in tqdm(queries, desc="Processing Queries"):
        # 1. Route query
        print("Routing query[{}]...".format(query['query']['query_id']))
        query_text = query['query']['content']
        prediction, doc_id = router(query, language) # prediction: domain, doc_id: list of doc_id
        print("prediction: {}".format(prediction))
        print("doc_id: {}".format(doc_id))
        # 2. Retrieve relevant chunks
        print("Retrieving chunks...")
        if (doc_id):
            retrieved_chunks = embedding_retriever(query_text, language, domain=prediction, doc_ids=doc_id)
        elif (prediction):
            retrieved_chunks = embedding_retriever(query_text, language, domain=prediction)
        else:
            retrieved_chunks = embedding_retriever(query_text, language, top_k=5)

        chunks = []
        for chunk in retrieved_chunks:
            chunks.append({"page_content": chunk})
        # 2. Retrieve relevant chunks
        # retrieved_chunks = retriever.retrieve(query_text, top_k=5)
        # 3. Generate Answer
        print("Generating answer...")
        answer = generate_answer(query_text, chunks)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = retrieved_chunks

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
