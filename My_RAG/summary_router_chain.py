import os
import json
from ollama import Client
from generator import load_ollama_config, load_prompts
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
DB_PATH = "db/dataset.db"

conn = Connection(DB_PATH)

def get_contents_from_db(target_doc_ids):
    target_docs = []
    target_set = set(target_doc_ids)

    for id in target_set:
        row = conn.execute("SELECT content FROM documents WHERE doc_id = ?", (id,))
        doc_content = row.fetchone()
        if doc_content:
            content_string = doc_content[0]
            target_docs.append(content_string)
    return target_docs

def generate_answer(query, context_chunks, language="en"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompts = load_prompts(type="summary_chain")
    if language not in prompts:
        print(f"Warning: Language '{language}' not found in prompts. Falling back to 'en'.")
        language = "en"

    prompt_template = prompts[language]
    prompt = prompt_template.format(query=query, context=context)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
        "num_ctx": 32768,
        # "temperature": 0.3,
        # "max_tokens": 1024,
        # "top_p": 0.9,
        # "top_k": 40,
        # "frequency_penalty": 0.1,
        # "presence_penalty": 0.1,
    }, prompt=prompt)
    return response["response"]

def summary_router_chain(query, language, doc_ids):
    query_text = query['query']['content']
    contents = get_contents_from_db(target_doc_ids=doc_ids)
    context = [{"page_content": content} for content in contents]
    
    raw_response = generate_answer(query_text, context, language)
    
    try:
        result_json = json.loads(raw_response)
        simple_list = result_json.get("retrieve", [])
        
        if isinstance(simple_list, str):
            simple_list = [simple_list]
        
        formatted_retrieve = []
        for text in simple_list:
            if isinstance(text, str):
                formatted_retrieve.append({"page_content": text})

        print("Generated Answer:", result_json)
        return result_json.get("answer", ""), formatted_retrieve
    
    except json.JSONDecodeError:
        print("JSON Parse Error. Raw content:", raw_response)
        return "Parsing Error", [{"page_content": raw_response}]
