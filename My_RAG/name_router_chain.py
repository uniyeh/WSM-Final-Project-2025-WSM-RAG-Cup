from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from utils import load_ollama_config
from ollama import Client
import ast
from Name_Router.generator import generate_combined_questions_answer, generate_compare_answer, generate_complex_answer, generate_medical_answer, generate_simple_answer
from Name_Router.reasoner import query_classifier, construct_multiple_questions, generate_sub_query_answer, question_and_answer_classifier, fallback_to_simple_check, compare_sub_questions_answer
from utils import DatasetConnection

def name_router_chain(query, language="en", prediction=None, doc_ids=[], doc_names=[]):
    query_text = query['query']['content']
    if (len(doc_ids) == 1):
        query_type = query_classifier(query_text, language)
        if ("COMPLEX" in query_type):
            answer, return_chunks = breakdown_path(query_text, language, prediction, doc_ids, doc_names)
            #fallback to check with simple answer
            if ("无法回答" not in answer and 'Unable to answer' not in answer):
                if (fallback_to_simple_check(query_text, answer, language, doc_names)):
                    return single_path(query_text, language, prediction, doc_ids, doc_names)
            return answer, return_chunks
        if (prediction == "Medical"):
            return single_medical_path(query_text, language, prediction, doc_ids, doc_names)
        return single_path(query_text, language, prediction, doc_ids, doc_names)
    else:
        return breakdown_path(query_text, language, prediction, doc_ids, doc_names)

def single_path(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    modified_query_text = get_remove_names_from_text(query_text, doc_names)

    # 1. Retrieve bigger chunks(use BM25)
    retrieved_chunks = []
    retrieved_chunks.extend(retrieve_bigger_chunks(query_text, language, prediction, doc_id, doc_names))
    # 2. Retrieve smaller chunks(use BM25)
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])


    # 3. Generate Answer
    answer = generate_simple_answer(query_text, return_chunks, language)
    if ("无法回答" in answer or 'Unable to answer' in answer):
        return answer, return_chunks
    
    #4. Fine-tune retriever
    retrieve_answer = get_remove_names_from_text(answer, doc_names)
    final_retrieve = modified_query_text + " " + retrieve_answer
    if (language == 'zh'):
        final_retrieve = retrieve_answer
    retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    return answer, return_chunks

def single_medical_path(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    #1. Retrieve bigger chunks(use BM25)
    conn = DatasetConnection()
    cursor = conn.execute("SELECT content FROM documents WHERE doc_id IN ({})".format(','.join(map(str, doc_id))))
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        docs.append({
            "content": row[0],
            "language": language
        })
    answer = generate_medical_answer(query_text, docs, language)
    retrieved_chunks = retrieve_bigger_chunks(query_text+ ' ' + answer, language, prediction, doc_id, doc_names)
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(answer, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    return answer, return_chunks

def single_complex_path(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    #1. Retrieve bigger chunks(use BM25)
    conn = DatasetConnection()
    cursor = conn.execute("SELECT content FROM documents WHERE doc_id IN ({})".format(','.join(map(str, doc_id))))
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        docs.append({
            "content": row[0],
            "language": language
        })
    combined_answer = generate_complex_answer(query_text, docs, language)
    retrieved_chunks = retrieve_bigger_chunks(query_text+ ' ' + combined_answer, language, prediction, doc_id, doc_names)
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    return combined_answer, return_chunks

def breakdown_path(query_text, language="en", prediction=None, doc_ids=[], doc_names=[]):
    queries = []
    return_sub_queries = construct_multiple_questions(query_text, language, doc_names)
    result = []
    return_chunks = []
    combined_answers = []
    combined_chunks = []
    # 1. return_sub_queries is now a list (or empty list on error)
    parsed_data = return_sub_queries

    if not parsed_data:
        print("[Breakdown Path] No valid questions generated or JSON error.")
        return single_complex_path(query_text, language, prediction, doc_ids, doc_names)

    # 2. Iterate through it easily
    for item in parsed_data:
        queries.append([item['doc_name'], item['sub_question']])

    # if (len(queries) < 2):
    #     return single_complex_path(query_text, language, prediction, doc_ids, doc_names)

    for sub_query_item in queries:
        doc_name = sub_query_item[0]
        sub_query = sub_query_item[1]
        modified_query_text = get_remove_names_from_text(sub_query, doc_names)

        single_doc_id = []
        conn = DatasetConnection()
        cursor = conn.execute("SELECT domain, name, doc_id FROM documents WHERE doc_id IN ({})".format(','.join(map(str, doc_ids))))
        rows = cursor.fetchall()
        #fallback to all documents if no document is found
        if (not rows):
            single_doc_id = doc_ids
        else:
            for row in rows:
                if (doc_name in row[1]):
                    single_doc_id.append(row[2])

        # 1. Retrieve bigger chunks(use BM25)
        retrieved_chunks = []
        retrieved_chunks.extend(retrieve_bigger_chunks(sub_query, language, prediction, single_doc_id, doc_names))
        
        if (language == 'en'):
            answer = generate_sub_query_answer(sub_query, retrieved_chunks, language)
            # 2. Retrieve smaller chunks(use BM25)
            small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
            query_text_for_small_retriever = modified_query_text
            if ("无法回答" not in answer and 'Unable to answer' not in answer):
                retrieve_answer = get_remove_names_from_text(answer, doc_names)
                query_text_for_small_retriever = modified_query_text + " " + retrieve_answer

            retriever_2 = create_retriever(small_retrieved_chunks, language)
            retrieved_small_chunks = retriever_2.retrieve(query_text_for_small_retriever, top1_check=True) # retrieve for higher than the top 1 score * 0.5
            return_chunks = []
            for index, chunk in enumerate(retrieved_small_chunks):
                return_chunks.append(small_chunks[chunk['chunk_index']])
        else:
            # # 2. Retrieve smaller chunks(use BM25)
            small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
            retriever_2 = create_retriever(small_retrieved_chunks, language)
            retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
            return_chunks = []
            for index, chunk in enumerate(retrieved_small_chunks):
                return_chunks.append(small_chunks[chunk['chunk_index']])

            # 3. Generate Answer
            answer = generate_sub_query_answer(sub_query, return_chunks, language)
            if ("无法回答" not in answer and 'Unable to answer' not in answer):
                #4. Fine-tune retriever
                retrieve_answer = get_remove_names_from_text(answer, doc_names)
                final_retrieve = retrieve_answer
                retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
                return_chunks = []
                for index, chunk in enumerate(retrieved_small_chunks):
                    return_chunks.append(small_chunks[chunk['chunk_index']])

        combined_chunks.extend(return_chunks)
        combined_answers.append(answer)

    # 3. Generate Final Answer
    if ('比较' in query_text or 'Compare' in query_text or 'compare' in query_text):
        compare_result = compare_sub_questions_answer(query_text, queries, combined_answers, combined_chunks, language)
        answer = generate_compare_answer(query_text, queries, combined_answers, combined_chunks, compare_result, language)
    else:
        answer = generate_combined_questions_answer(query_text, queries, combined_answers, combined_chunks, language)
    
    return answer, combined_chunks

########## Helper Functions ##########

def retrieve_bigger_chunks(query, language="en", prediction=None, doc_id=[], doc_names=[]):
    row_chunks = get_chunks_from_db(prediction, doc_id, language)
    retriever = create_retriever(row_chunks, language)
    
    retrieved_chunks = retriever.retrieve(query, threshold=0) # retrieve as much as possible
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

def breakdown_combine_chunks(chunks, language="en"):
    return_chunks = []
    for chunk in chunks:
        new_chunks = chunk['page_content'].split("\n")
        for new_chunk in new_chunks:
            item = chunk.copy()
            item['page_content'] = new_chunk
            return_chunks.append(item)
    return return_chunks