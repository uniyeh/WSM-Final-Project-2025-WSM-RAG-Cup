import re
import numpy as np
from ollama import Client
import os
import sys
from name_router_chain import name_router_chain
from default_chain import default_chain
from llm_router_chain import llm_router_chain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../db')))
from Connection import Connection
DB_PATH = "db/dataset.db"

# TODO: to be implemented
def is_summary_router(query, language):
    return False

def router(query, language="en"):
    ## Step 0. keywords matching
    prediction, doc_id, matched_name = name_matcher(query, language)
    # time_matcher(query, language)
    # name_matcher(query, language)
    print("[Router] matching result: ", prediction, doc_id, matched_name)

    ## Step 1. summary chain (TODO)
    # print("[Router][1] summary chain")
    # if (is_summary_router(query, language)):
    #     return summary_router_chain(query, language, prediction, doc_id, matched_name)
    
    ## Step 2. name_router chain
    if (prediction):
        print("[Router][2] name_router chain")
        return name_router_chain(query, language, prediction, doc_id, matched_name)
    
    ## Step 3. LLM chain (TODO)
    print("[Router][3] LLM chain")
    return llm_router_chain(query, language)

    ## Step 4. fallback to old default chain
    # print("[Router][4] fallback to old default chain")
    # return default_chain(query, language)

def name_matcher(query, language="en"):
    content = query['query']['content']
    prediction = None
    doc_id = []
    matched_name = []

    # string matching logic
    conn = Connection(DB_PATH)
    rows = conn.execute("SELECT domain, name, doc_id FROM documents WHERE language = ?", (language,))
    name_docs = {}
    for row in rows:
        domain = row[0]
        name = row[1]
        if name not in name_docs:
            name_docs[name] = {
                'doc_id': [],
                'domain': domain
            }
        name_docs[name]['doc_id'].append(row[2])

    for name in name_docs:
        if (name_docs[name]['domain'] == 'Law'):
            match = False
            if (language == 'en'):
                split_name = name.split(',')
                match = True
                for i in range(0, len(split_name)):
                    if split_name[i] not in content:
                        match = False
                        break
            else:
                match = name in content
            if (match):
                prediction = 'Law'
                doc_id.extend(name_docs[name]['doc_id'])
                matched_name.append(name)
        elif (name_docs[name]['domain'] == 'Medical'):
            hospital_name = name.split("_")[0]
            if (hospital_name in content):
                prediction = 'Medical'
                doc_id.extend(name_docs[name]['doc_id'])
                matched_name.append(name)
        elif (name_docs[name]['domain'] == 'Finance'):
            if (name in content):
                prediction = 'Finance'
                doc_id.extend(name_docs[name]['doc_id']) 
                matched_name.append(name)
    if (prediction):
        if (prediction == 'Medical'):
            if (len(matched_name) >= 1):
                new_doc_id = []
                new_matched_name = []
                for name in matched_name:
                    hospital_name = name.split("_")[0]
                    user_name = name.split("_")[1]
                    if (user_name in content):
                        new_doc_id.extend(name_docs[name]['doc_id'])
                        new_matched_name.append(hospital_name)
                if (new_doc_id):
                    doc_id = new_doc_id
                    matched_name = new_matched_name
                else:
                    new_matched_name = []
                    for name in matched_name:
                        hospital_name = name.split("_")[0]
                        new_matched_name.append(hospital_name)
                    matched_name = new_matched_name
        return prediction, doc_id, matched_name

    # # Try LLM-based matching
    # """
    # Use LLM to intelligently match query to document names.
    # """
    # from subject_matcher import find_doc_names
    # from router_utils import cache_document_names    
    # try:
    #     matched_names = find_doc_names(content, language=language, top_k=3)
        
    #     if matched_names:
    #         # Get the document cache to look up doc_ids and domain
    #         doc_cache = cache_document_names(language)
    #         for name in matched_names:
    #             if name in doc_cache:
    #                 doc_id.extend(doc_cache[name]['doc_ids'])
    #                 # Use the domain of the first matched document
    #                 if prediction is None:
    #                     prediction = doc_cache[name]['domain']
            
    #         if doc_id:
    #             print(f"✓ LLM name_router matched: {matched_names}")
    #             return prediction, doc_id, matched_names
    # except Exception as e:
    #     print(f"LLM name_router failed: {e}, falling back to string matching")
    
    return None, [], []
