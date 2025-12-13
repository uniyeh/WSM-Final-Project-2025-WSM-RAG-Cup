import sqlite3
from generator import generate_answer
from retriever import get_chunks_from_db, create_retriever
from entity_extractor import extract_entities
from runtime_chunker import chunk_row_chunks

DB_PATH = "db/dataset.db"

def time_router_chain(query, language, doc_ids=[]):
    """
    Time-based router chain that filters chunks by temporal entities.
    Uses multi-step retrieval similar to name_router_chain.
    
    Args:
        query: Query dictionary with query content
        language: Language code ('en' or 'zh')
        doc_ids: List of document IDs to filter by
        
    Returns:
        Tuple of (answer, chunks)
    """
    query_text = query['query']['content']
    
    # Extract temporal entities from query
    entities = extract_entities(query_text, language, use_llm=False)
    
    print(f"[TimeRouter] Extracted entities: years={entities['years']}, months={entities['months']}")
    
    # Step 1: Get chunks filtered by time entities (excluding Finance domain first)
    print("[TimeRouter][1] Retrieve with time filter (excluding Finance):")
    retrieved_chunks = get_chunks_with_time_filter(doc_ids, language, entities, exclude_finance=True)
    
    if not retrieved_chunks:
        print("[TimeRouter] No non-Finance chunks found, trying with all domains...")
        retrieved_chunks = get_chunks_with_time_filter(doc_ids, language, entities, exclude_finance=False)
        
        if not retrieved_chunks:
            print("[TimeRouter] No chunks found with time filter, falling back to all chunks")
            retrieved_chunks = get_chunks_from_db(None, doc_ids, language)
    
    print(f"[TimeRouter] Retrieved {len(retrieved_chunks)} bigger chunks")
    
    # Step 2: Create smaller chunks for more precise retrieval
    print("[TimeRouter][2] Create smaller chunks:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks(language, retrieved_chunks)
    
    # Step 3: Retrieve smaller chunks using BM25
    print("[TimeRouter][3] Retrieve smaller chunks with BM25:")
    retriever = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever.retrieve(query_text, top1_check=True)
    
    return_chunks = []
    for chunk in retrieved_small_chunks:
        return_chunks.append(small_chunks[chunk['chunk_index']])
    
    print(f"[TimeRouter] Retrieved {len(return_chunks)} smaller chunks")
    
    # Step 4: Generate answer
    print("[TimeRouter][4] Generate answer:")
    answer = generate_answer(query_text, return_chunks, language, type="llm_chain")
    
    # Step 5: Fine-tune retrieval based on answer (if answer is valid)
    if "无法回答" not in answer and "Unable to answer" not in answer:
        print("[TimeRouter][5] Fine-tune retrieval based on answer:")
        final_retrieve = query_text + " " + answer
        if language == 'zh':
            final_retrieve = answer
        
        retrieved_small_chunks = retriever.retrieve(final_retrieve, top1_check=True)
        return_chunks = []
        for chunk in retrieved_small_chunks:
            return_chunks.append(small_chunks[chunk['chunk_index']])
        
        print(f"[TimeRouter] Final chunks: {len(return_chunks)}")
    
    return answer, return_chunks


########## Helper Functions ##########

def create_smaller_chunks(language="en", retrieved_chunks=[]):
    """
    Create smaller chunks from retrieved chunks for more precise retrieval.
    Similar to name_router_chain's create_smaller_chunks_without_names.
    
    Args:
        language: Language code
        retrieved_chunks: List of bigger chunks
        
    Returns:
        Tuple of (small_retrieved_chunks, small_chunks)
    """
    small_chunks = chunk_row_chunks(retrieved_chunks, language)
    small_retrieved_chunks = []
    for index, chunk in enumerate(small_chunks):
        small_retrieved_chunks.append({
            "page_content": chunk['page_content'],
            "chunk_index": index
        })
    return small_retrieved_chunks, small_chunks


def filter_out_company_chunks(chunks, query_text, language):
    """
    Filter out chunks that contain specific company names mentioned in the query.
    These should be handled by name_router instead.
    
    Args:
        chunks: List of chunks
        query_text: Original query text
        language: Language code
        
    Returns:
        Filtered list of chunks
    """
    # Extract company names from query
    entities = extract_entities(query_text, language, use_llm=False)
    company_names = entities.get('companies', [])
    
    if not company_names:
        return chunks  # No company names to filter
    
    print(f"[TimeRouter] Filtering out chunks with companies: {company_names}")
    
    # Filter out chunks that contain any of the company names
    filtered_chunks = []
    for chunk in chunks:
        content = chunk.get('page_content', '')
        # Check if chunk contains any company name
        has_company = False
        for company in company_names:
            if company in content:
                has_company = True
                break
        
        if not has_company:
            filtered_chunks.append(chunk)
    
    print(f"[TimeRouter] Filtered {len(chunks)} → {len(filtered_chunks)} chunks (removed {len(chunks) - len(filtered_chunks)} with company names)")
    
    return filtered_chunks


def get_chunks_with_time_filter(doc_ids, language, entities, exclude_finance=True):
    """
    Get chunks filtered by temporal entities, optionally excluding Finance domain chunks.
    
    Args:
        doc_ids: List of document IDs
        language: Language code
        entities: Dictionary with extracted entities (years, months, dates, people)
        exclude_finance: If True, exclude Finance domain documents (default: True)
        
    Returns:
        List of chunks matching the time filters
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Build WHERE clause
    where_clauses = []
    params = []
    
    # Language filter
    where_clauses.append("chunks.language = ?")
    params.append(language)
    
    # Document ID filter
    if doc_ids:
        placeholders = ','.join('?' for _ in doc_ids)
        where_clauses.append(f"chunks.doc_id IN ({placeholders})")
        params.extend(doc_ids)
    
    # Time filters (AND condition - must match ALL specified time entities)
    if entities['years']:
        year_conditions = []
        for year in entities['years']:
            year_conditions.append("chunks.years LIKE ?")
            params.append(f"%{year}%")
        # If multiple years, use OR for years
        if year_conditions:
            where_clauses.append(f"({' OR '.join(year_conditions)})")
    
    if entities['months']:
        month_conditions = []
        for month in entities['months']:
            month_conditions.append("chunks.months LIKE ?")
            params.append(f"%{month}%")
        # If multiple months, use OR for months
        if month_conditions:
            where_clauses.append(f"({' OR '.join(month_conditions)})")
    
    # Optionally exclude chunks from documents with company names (Finance domain)
    # These should be handled by name_router
    if exclude_finance:
        where_clauses.append("documents.domain != 'Finance'")
    
    # Build final query with JOIN to documents table
    where_clause = " AND ".join(where_clauses)
    query = f"""
        SELECT chunks.id, chunks.name, chunks.content 
        FROM chunks 
        JOIN documents ON chunks.doc_id = documents.doc_id AND chunks.language = documents.language
        WHERE {where_clause}
    """
    
    print(f"[TimeRouter] SQL: {query}")
    print(f"[TimeRouter] Params: {params}")
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to chunk format
    chunks = []
    for row in rows:
        chunks.append({
            "id": row[0],
            "name": row[1],
            "page_content": row[2]
        })
    
    return chunks

