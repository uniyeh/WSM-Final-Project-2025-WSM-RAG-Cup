#!/usr/bin/env python3
"""
Rebuild database with entity extraction.
This script:
1. Reads existing documents and chunks
2. Extracts entities (years, months, dates, people) from content
3. Creates new database with entity fields populated
"""

import sqlite3
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from My_RAG.entity_extractor import extract_entities

DB_PATH_OLD = "db/dataset.db"
DB_PATH_NEW = "db/dataset_v2.db"

def create_new_schema(conn):
    """Create new database schema with entity fields."""
    cursor = conn.cursor()
    
    # Create documents table with entity fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            language TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            jsonl TEXT NOT NULL,
            years TEXT,
            months TEXT,
            dates TEXT,
            people TEXT
        )
    """)
    
    # Create chunks table with entity fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            language TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            years TEXT,
            months TEXT,
            dates TEXT,
            people TEXT
        )
    """)
    
    # Create indexes for faster entity queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_years ON chunks(years)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_people ON chunks(people)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_years ON documents(years)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_people ON documents(people)")
    
    conn.commit()
    print("✓ Created new schema with entity fields")

def migrate_documents(old_conn, new_conn):
    """Migrate documents table with entity extraction."""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Read all documents
    old_cursor.execute("SELECT id, doc_id, domain, language, name, content, jsonl FROM documents")
    documents = old_cursor.fetchall()
    
    print(f"\nMigrating {len(documents)} documents...")
    
    for i, (id, doc_id, domain, language, name, content, jsonl) in enumerate(documents):
        # Extract entities from content
        entities = extract_entities(content, language, use_llm=False)
        
        # Convert lists to comma-separated strings
        years_str = ','.join(entities['years']) if entities['years'] else None
        months_str = ','.join(entities['months']) if entities['months'] else None
        dates_str = ','.join(entities['dates']) if entities['dates'] else None
        people_str = ','.join(entities['people']) if entities['people'] else None
        
        # Insert into new database
        new_cursor.execute("""
            INSERT INTO documents (id, doc_id, domain, language, name, content, jsonl, years, months, dates, people)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, doc_id, domain, language, name, content, jsonl, years_str, months_str, dates_str, people_str))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(documents)} documents...")
    
    new_conn.commit()
    print(f"✓ Migrated {len(documents)} documents")

def migrate_chunks(old_conn, new_conn):
    """Migrate chunks table with entity extraction."""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    # Read all chunks
    old_cursor.execute("SELECT id, doc_id, domain, language, name, content FROM chunks")
    chunks = old_cursor.fetchall()
    
    print(f"\nMigrating {len(chunks)} chunks...")
    
    for i, (id, doc_id, domain, language, name, content) in enumerate(chunks):
        # Extract entities from content
        entities = extract_entities(content, language, use_llm=False)
        
        # Convert lists to comma-separated strings
        years_str = ','.join(entities['years']) if entities['years'] else None
        months_str = ','.join(entities['months']) if entities['months'] else None
        dates_str = ','.join(entities['dates']) if entities['dates'] else None
        people_str = ','.join(entities['people']) if entities['people'] else None
        
        # Insert into new database
        new_cursor.execute("""
            INSERT INTO chunks (id, doc_id, domain, language, name, content, years, months, dates, people)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, doc_id, domain, language, name, content, years_str, months_str, dates_str, people_str))
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks...")
    
    new_conn.commit()
    print(f"✓ Migrated {len(chunks)} chunks")

def verify_migration(conn):
    """Verify the migration was successful."""
    cursor = conn.cursor()
    
    print("\n=== Migration Verification ===")
    
    # Count documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"Documents: {doc_count}")
    
    # Count chunks
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"Chunks: {chunk_count}")
    
    # Show sample with entities
    print("\n=== Sample Chunks with Entities ===")
    cursor.execute("""
        SELECT name, years, months, people, SUBSTR(content, 1, 100) as content_preview
        FROM chunks 
        WHERE years IS NOT NULL OR people IS NOT NULL
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        name, years, months, people, content = row
        print(f"\nChunk: {name}")
        print(f"  Years: {years}")
        print(f"  Months: {months}")
        print(f"  People: {people}")
        print(f"  Content: {content}...")
    
    # Statistics
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE years IS NOT NULL")
    chunks_with_years = cursor.fetchone()[0]
    print(f"\n✓ Chunks with years: {chunks_with_years} ({chunks_with_years/chunk_count*100:.1f}%)")
    
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE people IS NOT NULL")
    chunks_with_people = cursor.fetchone()[0]
    print(f"✓ Chunks with people: {chunks_with_people} ({chunks_with_people/chunk_count*100:.1f}%)")

def main():
    print("="*60)
    print("Database Migration: Adding Entity Fields")
    print("="*60)
    
    # Check if old database exists
    if not os.path.exists(DB_PATH_OLD):
        print(f"Error: Old database not found at {DB_PATH_OLD}")
        return
    
    # Remove new database if it exists
    if os.path.exists(DB_PATH_NEW):
        print(f"Removing existing {DB_PATH_NEW}...")
        os.remove(DB_PATH_NEW)
    
    # Connect to databases
    print(f"\nConnecting to databases...")
    old_conn = sqlite3.connect(DB_PATH_OLD)
    new_conn = sqlite3.connect(DB_PATH_NEW)
    
    try:
        # Create new schema
        create_new_schema(new_conn)
        
        # Migrate data
        migrate_documents(old_conn, new_conn)
        migrate_chunks(old_conn, new_conn)
        
        # Verify migration
        verify_migration(new_conn)
        
        print("\n" + "="*60)
        print("✓ Migration completed successfully!")
        print(f"✓ New database created: {DB_PATH_NEW}")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Review the new database")
        print("2. Backup your old database:")
        print(f"   cp {DB_PATH_OLD} {DB_PATH_OLD}.backup")
        print("3. Replace old database with new one:")
        print(f"   mv {DB_PATH_NEW} {DB_PATH_OLD}")
        
    finally:
        old_conn.close()
        new_conn.close()

if __name__ == "__main__":
    main()
