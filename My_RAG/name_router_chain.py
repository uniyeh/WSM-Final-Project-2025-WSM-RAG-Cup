from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from utils import load_ollama_config
from ollama import Client
import ast
from generator import generate_answer
import json
from name_router_chain_generator import generate_sub_query_answer, generate_combined_questions_answer, construct_multiple_questions, compare_then_generate_answer, query_classifier, generate_complex_answer
import sqlite3
import os
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../db/dataset.db'))

def name_router_chain(query, language="en", prediction=None, doc_ids=[], doc_names=[]):
    query_text = query['query']['content']
    if (len(doc_ids) == 1):
        query_type = query_classifier(query_text, language)
        if ("COMPLEX" in query_type):
            print("[Single Path-COMPLEX] query_text: ", query_text)
            return single_complex_path(query_text, language, prediction, doc_ids, doc_names)
        return single_path(query_text, language, prediction, doc_ids, doc_names)
    else:
        return breakdown_path(query_text, language, prediction, doc_ids, doc_names)

def single_path(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    print("[Single Path] query_text: ", query_text)
    modified_query_text = get_remove_names_from_text(query_text, doc_names)

    # 1. Retrieve bigger chunks(use BM25)
    print("[1] retrieve with bigger chunks:")
    retrieved_chunks = []
    retrieved_chunks.extend(retrieve_bigger_chunks(query_text, language, prediction, doc_id, doc_names))
    print("retrieved_chunks: ", retrieved_chunks)
    # 2. Retrieve smaller chunks(use BM25)
    print("[2] retrieve with smaller chunks and extract document name:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    print("retrieved_small_chunks: ", retrieved_small_chunks)
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])

    print('chunks: ', len(return_chunks))

    # 3. Generate Answer
    print("[3] generate answer:")
    answer = generate_answer(query_text, return_chunks, language)
    print("answer: ", answer)
    if ("无法回答" in answer or 'Unable to answer' in answer):
        print('Unable to answer: \n')
        retrieved_small_chunks = retriever_2.retrieve(modified_query_text, threshold=0.0)
        return_chunks = []
        for index, chunk in enumerate(retrieved_small_chunks):
            return_chunks.append(small_chunks[chunk['chunk_index']])
        print('try again chunks: ', len(return_chunks))
        answer = generate_answer(query_text, return_chunks, language)
        print("try again check for the Unable to answer : ", answer)
        if ("无法回答" in answer or 'Unable to answer' in answer):
            return answer, return_chunks
    
    #4. Fine-tune retriever
    retrieve_answer = get_remove_names_from_text(answer, doc_names)
    final_retrieve = modified_query_text + " " + retrieve_answer
    if (language == 'zh'):
        final_retrieve = retrieve_answer
    print("[4] rerieve for final answer: {}".format(final_retrieve))
    retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    print('final chunks: ', len(return_chunks))
    return answer, return_chunks


def single_complex_path(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    #1. Retrieve bigger chunks(use BM25)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM documents WHERE doc_id IN ({})".format(','.join(map(str, doc_id))))
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        docs.append({
            "content": row[0],
            "language": language
        })
    conn.close()
    combined_answer = generate_complex_answer(query_text, docs, language)
    retrieved_chunks = retrieve_bigger_chunks(query_text+ ' ' + combined_answer, language, prediction, doc_id, doc_names)
    print("[4] retrieve with smaller chunks and extract document name:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    print('chunks: ', len(return_chunks))
    return combined_answer, return_chunks

def breakdown_path(query_text, language="en", prediction=None, doc_ids=[], doc_names=[]):
    queries = []
    return_sub_queries = construct_multiple_questions(query_text, language, doc_names)
    result = []
    return_chunks = []
    combined_answers = []
    combined_chunks = []
    try:
        # 1. Parse the string into a real Python list
        parsed_data = json.loads(return_sub_queries)

        # 2. Iterate through it easily
        for item in parsed_data:
            queries.append([item['doc_name'], item['sub_question']])

    except json.JSONDecodeError:
        print("[Breakdown Path] The LLM output was not valid JSON.")
        return single_complex_path(query_text, language, prediction, doc_ids, doc_names)

    if (len(queries) < 2):
        return single_complex_path(query_text, language, prediction, doc_ids, doc_names)

    print("[Breakdown Path] queries: ")
    for sub_query_item in queries:
        doc_name = sub_query_item[0]
        sub_query = sub_query_item[1]
        print("sub_query: ", sub_query)

        single_doc_id = []
        for index, name in enumerate(doc_names):
            if name == doc_name:
                single_doc_id.append(doc_ids[index])
        modified_query_text = get_remove_names_from_text(sub_query, doc_names)

        #fallback to all documents if no document is found
        if (not single_doc_id):
            single_doc_id = doc_ids

        # 1. Retrieve bigger chunks(use BM25)
        print("[1] retrieve with bigger chunks:")
        retrieved_chunks = []
        retrieved_chunks.extend(retrieve_bigger_chunks(sub_query, language, prediction, single_doc_id, doc_names))
        answer = generate_sub_query_answer(sub_query, retrieved_chunks, language)

        # 2. Retrieve smaller chunks(use BM25)
        print("[2] retrieve with smaller chunks and extract document name:")
        small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
        query_text_for_small_retriever = modified_query_text
        if ("无法回答" not in answer or 'Unable to answer' not in answer):
            retrieve_answer = get_remove_names_from_text(answer, doc_names)
            query_text_for_small_retriever = modified_query_text + " " + retrieve_answer

        retriever_2 = create_retriever(small_retrieved_chunks, language)
        retrieved_small_chunks = retriever_2.retrieve(query_text_for_small_retriever, top1_check=True) # retrieve for higher than the top 1 score * 0.5
        return_chunks = []
        for index, chunk in enumerate(retrieved_small_chunks):
            return_chunks.append(small_chunks[chunk['chunk_index']])

        # # 3. Generate Answer
        # print("[3] generate answer:")
        # # answer = generate_sub_query_answer(sub_query, return_chunks, language)
        # if ("无法回答" not in answer or 'Unable to answer' not in answer):
        #     #4. Fine-tune retriever
        #     retrieve_answer = get_remove_names_from_text(answer, doc_names)
        #     final_retrieve = modified_query_text + " " + retrieve_answer
        #     print("[4] rerieve for final answer: {}".format(final_retrieve))
        #     retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
        #     return_chunks = []
        #     for index, chunk in enumerate(retrieved_small_chunks):
        #         return_chunks.append(small_chunks[chunk['chunk_index']])
        #     print('final chunks: ', len(return_chunks))

        combined_chunks.extend(return_chunks)
        combined_answers.append(answer)

    # 3. Generate Final Answer
    print("[4] generate final answer:")
    if ('比较' in query_text or 'Compare' in query_text or 'compare' in query_text):
        answer = compare_then_generate_answer(query_text, queries, combined_answers, combined_chunks, language)
    else:
        answer = generate_combined_questions_answer(query_text, queries, combined_answers, combined_chunks, language)
    
    # Test for without fine-tune retrieve
    return answer, combined_chunks

    if ("无法回答" in answer or 'Unable to answer' in answer):
        return answer, combined_chunks
    
    if (not combined_chunks):
        return answer, combined_chunks
    if (language == 'zh'):
        return answer, combined_chunks
    #4. Fine-tune retriever
    retriever_final = create_retriever(combined_chunks, language)
    final_chunks = retriever_final.retrieve(answer, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    print('final return chunks:', len(final_chunks))
    return answer, final_chunks

########## Helper Functions ##########

def retrieve_bigger_chunks(query, language="en", prediction=None, doc_id=[], doc_names=[]):
    row_chunks = get_chunks_from_db(prediction, doc_id, language)
    retriever = create_retriever(row_chunks, language)
    
    retrieved_chunks = retriever.retrieve(query, threshold=0) # retrieve as much as possible
    print('chunks: ', len(retrieved_chunks))
    return retrieved_chunks

def create_smaller_chunks_without_names(language="en", retrieved_chunks=[], doc_names=[]):
    small_chunks = chunk_row_chunks(retrieved_chunks, language)
    small_retrieved_chunks = []
    for index, chunk in enumerate(small_chunks):
        small_retrieved_chunks.append({
            "page_content": get_remove_names_from_text(chunk['page_content'], doc_names),
            "chunk_index": index
        })
    return small_retrieved_chunks, small_chunks

def get_remove_names_from_text(content, doc_names = []):
    if (doc_names):
        for doc_name in doc_names:
            content = content.replace(doc_name, "")
    return content