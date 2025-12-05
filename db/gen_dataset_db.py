import jsonlines
import json
from Connection import Connection
from chunker import single_chunk
from utils import create_table_from_yaml

SCHEMA_PATH = 'db/dataset_table-schema.yaml'
DB_PATH = 'db/dataset.db'
DATASET_PATH = 'dragonball_dataset/dragonball_docs.jsonl'
SPECIAL_DATASET_PATH = 'db/special_dataset.jsonl'

def create_tables():
    conn = Connection(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute("DROP TABLE IF EXISTS chunks")
    create_table_from_yaml(SCHEMA_PATH, DB_PATH)

def main(docs_path):
    create_tables()
    populate_documents(docs_path)
    insert_special_documents()
    populate_chunks(docs_path)
    insert_special_chunks()

##################################
# Here for documents table
##################################

def insert_document(doc):
    conn = Connection(DB_PATH)
    domain = doc['domain']
    mapping = {
        "Finance": "company_name",
        "Law": "court_name",
        "Medical": "hospital_patient_name"
    }
    name = doc[mapping[domain]]
    if domain == "Medical":
        name = name.split("_")[0]
    conn.execute(
        "INSERT INTO documents (doc_id, domain, language, name, content, jsonl) VALUES (?, ?, ?, ?, ?, ?)",
        (doc['doc_id'], domain, doc['language'], name, doc['content'], json.dumps(doc))
    )
        

def populate_documents(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        for doc in reader:
            insert_document(doc)

##################################
# Here for chunks table
##################################

def insert_chunks(doc):
    conn = Connection(DB_PATH)
    chunks = single_chunk(doc['content'])
    mapping = {
        "Finance": "company_name",
        "Law": "court_name",
        "Medical": "hospital_patient_name"
    }
    name = doc[mapping[doc['domain']]]
    if doc['domain'] == "Medical":
        name = name.split("_")[0]
    for chunk in chunks:
        conn.execute(
            "INSERT INTO chunks (doc_id, domain, language, name, content) VALUES (?, ?, ?, ?, ?)",
            (doc['doc_id'], doc['domain'], doc['language'], name, chunk['page_content'])
        )

def populate_chunks(file_path=DATASET_PATH):
    with jsonlines.open(file_path, 'r') as reader:
        for doc in reader:
            insert_chunks(doc)

##################################
# Here for handling modified special dataset
##################################

def insert_special_documents():
    with jsonlines.open(SPECIAL_DATASET_PATH, 'r') as reader:
        for doc in reader:
            insert_document(doc)

def insert_special_chunks():
    with jsonlines.open(SPECIAL_DATASET_PATH, 'r') as reader:
        for chunk in reader:
            insert_chunks(chunk)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen', type=bool, default=False, help='Regenerate the database [default: False]')
    parser.add_argument('--docs_path', type=str, default=DATASET_PATH, help='Path to the documents file')
    args = parser.parse_args()
    if (args.regen):
        main(args.docs_path)
    else:
        print("No action taken.")