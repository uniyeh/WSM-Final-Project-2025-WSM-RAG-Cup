import jsonlines
import json
from Connection import Connection
from chunker import single_chunk
from utils import create_table_from_yaml

SCHEMA_PATH = 'db/query_table-schema.yaml'
DB_PATH = 'db/dataset.db'
DATASET_PATH = 'dragonball_dataset/dragonball_queries.jsonl'

def create_tables():
    conn = Connection(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS queries")
    create_table_from_yaml(SCHEMA_PATH, DB_PATH)

def main(docs_path):
    create_tables()
    populate_queries(docs_path)

##################################
# Here for queries table
##################################

def insert_query(doc):
    conn = Connection(DB_PATH)
    query = doc['query']
    ground_truth = doc['ground_truth']
    references = "\n".join(ground_truth['references']) if ground_truth['references'] else ""
    conn.execute(
        "INSERT INTO queries (query_id, domain, query_type, language, query, answer, doc_count, refs, jsonl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (query['query_id'], doc['domain'], query['query_type'], doc['language'], query['content'], ground_truth['content'], len(ground_truth['doc_ids']), references, json.dumps(doc))
    )

def populate_queries(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        for doc in reader:
            insert_query(doc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen', type=bool, default=False, help='Regenerate the database [default: False]')
    parser.add_argument('--docs_path', type=str, default=DATASET_PATH, help='Path to the queries file')
    args = parser.parse_args()
    if (args.regen):
        main(args.docs_path)
    else:
        print("No action taken.")