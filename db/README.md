# Database
**The database is used to store the processed dataset and the training queries.**
- We use sqlite3 as the database.
- The database is stored in `db/`.

=================================================

## Regenerate the database
**Regenerate the dataset db (only "documents" and "chunks" tables)**
To regenerate the dataset db, run the following command:
```bash
./db/run_setting.sh
```
=================================================

## Database
### store in `db/dataset.db`
1. "documents" and "chunks" tables from `dragonball_dataset/dragonball_docs.jsonl`
**Here we store the processed dataset, which contains "documents" and "chunks" tables.**
- The dataset is stored in `db/dataset.db`.
- The schema is stored in `dataset_table-schema.yaml`.

2. "queries" tables from `dragonball_dataset/dragonball_queries.jsonl`
**Here we store the training queries.**
- The query is stored in `db/dataset.db`.
- The schema is stored in `query_table-schema.yaml`.

=================================================
## faiss
**Here save the embedding vector from "qwen3-embedding:0.6b"**
*** or "embeddinggemma:300m"? ***

### use faiss (TBD)
*** `My_RAG/embedding_retriever.py` or `My_RAG/router.py` ***

### 1. documents
**Here we store the faiss index for documents.**
```bash
python db/faiss/documents/generate_faiss.py
```

### 2. chunks
**Here we store the faiss index for chunks from documents.**
```bash
python db/faiss/chunks/generate_faiss.py
```

### queries
**He3. re we store the faiss index for queries.**
```bash
python db/faiss/queries/generate_faiss.py
```
