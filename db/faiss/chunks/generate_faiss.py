import faiss
import numpy as np
import os
import sys
import shutil
import json
from ollama import Client
from tqdm import tqdm

# Add project root to path to import My_RAG and db
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from My_RAG.utils import load_ollama_config
from db.Connection import Connection

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset.db'))

def generate_vector(text):
    # Generate vector using Ollama embeddinggemma:300m or qwen3-embedding:0.6b
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.embeddings(model="qwen3-embedding:0.6b", prompt=text)
    return np.array(response['embedding'], dtype='float32').reshape(1, -1)

def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def generate_faiss_index(language, output_dir):
    clear_and_create_dir(output_dir)
    conn = Connection(DB_PATH)
    # Use language column instead of domain
    cursor = conn.execute("SELECT id, content FROM chunks WHERE language = ?", (language,))
    docs = cursor.fetchall()
    
    print(f"Generating {language} indices in {output_dir}...")
    
    vectors = []
    ids = []
    
    expected_dim = None
    
    for doc in tqdm(docs):
        id = doc[0]
        content = doc[1]
        try:
            vector = generate_vector(content)
            current_dim = vector.shape[1]
            
            if expected_dim is None:
                expected_dim = current_dim
            elif current_dim != expected_dim:
                print(f"Skipping id {id}: Vector dimension mismatch. Expected {expected_dim}, got {current_dim}")
                continue
                
            vectors.append(vector[0])
            ids.append(id)
        except Exception as e:
            print(f"Error processing id {id}: {e}")
            
    if not vectors:
        print(f"No documents found for language: {language}")
        return

    # Convert to numpy array
    vectors_np = np.array(vectors).astype('float32')
    
    # Create FAISS index
    d = vectors_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors_np)
    
    # Save index
    index_path = os.path.join(output_dir, f"{language}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    # Save mapping (FAISS ID -> Doc ID)
    # FAISS IDs are just 0, 1, 2... corresponding to the order of addition
    mapping = {i: id for i, id in enumerate(ids)}
    mapping_path = os.path.join(output_dir, f"{language}_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f)
    print(f"Saved mapping to {mapping_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    generate_faiss_index('en', os.path.join(base_dir, 'en'))
    generate_faiss_index('zh', os.path.join(base_dir, 'zh'))
    