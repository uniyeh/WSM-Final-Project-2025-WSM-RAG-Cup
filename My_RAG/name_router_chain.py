from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from utils import load_ollama_config
from ollama import Client
import ast
from generator import generate_answer

def retrieve_bigger_chunks(query, language="en", prediction=None, doc_id=[], doc_names=[]):
    row_chunks = get_chunks_from_db(prediction, doc_id, language)
    retriever = create_retriever(row_chunks, language)
    
    retrieved_chunks = retriever.retrieve(query, threshold=0) # retrieve as much as possible
    print('chunks: ', len(retrieved_chunks))
    return retrieved_chunks

def name_router_chain(query, language="en", prediction=None, doc_ids=[], doc_names=[]):
    query_text = query['query']['content']
    if (len(doc_ids) == 1):
        return retrieve_single_query(query_text, language, prediction, doc_ids, doc_names)

    else:
        queries = construct_multiple_questions(query_text, language, doc_names)
        result = []
        return_chunks = []
        if (not queries):
            print("[name_router_chain] no queries")
            return retrieve_single_query(query_text, language, prediction, doc_ids, doc_names)
        
        combined_answers = []
        combined_chunks = []
        for sub_query_item in queries:
            doc_name = sub_query_item[0]
            sub_query = sub_query_item[1]
            for index, name in enumerate(doc_names):
                if name == doc_name:
                    doc_id = doc_ids[index]
            modified_query_text = get_remove_names_from_text(sub_query, doc_names)
            print("sub_query: ", sub_query)
            print("modified_query_text: ", modified_query_text)

            # 1. Retrieve bigger chunks(use BM25)
            print("[1] retrieve with bigger chunks:")
            retrieved_chunks = []
            retrieved_chunks.extend(retrieve_bigger_chunks(sub_query, language, prediction, [doc_id], doc_names))

            # 2. Retrieve smaller chunks(use BM25)
            print("[2] retrieve with smaller chunks and extract document name:")
            small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
            
            retriever_2 = create_retriever(small_retrieved_chunks, language)
            retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
            return_chunks = []
            for index, chunk in enumerate(retrieved_small_chunks):
                return_chunks.append(small_chunks[chunk['chunk_index']])

            print('return_chunks: ', return_chunks)

            # 3. Generate Answer
            print("[3] generate answer:")
            answer = generate_sub_query_answer(sub_query, return_chunks, language)
            if ("无法回答" not in answer and 'Unable to answer' not in answer):
                #4. Fine-tune retriever
                retrieve_answer = get_remove_names_from_text(answer, doc_names)
                final_retrieve = modified_query_text + " " + retrieve_answer
                print("[4] rerieve for final answer: {}".format(final_retrieve))
                retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
                return_chunks = []
                for index, chunk in enumerate(retrieved_small_chunks):
                    return_chunks.append(small_chunks[chunk['chunk_index']])
                print('final chunks: ', len(return_chunks))

            combined_chunks.extend(return_chunks)
            combined_answers.append(answer)
        print("result: ", result)

        # 3. Generate Final Answer
        print("[4] generate final answer:")
        answer = generate_combined_questions_answer(query_text, queries, combined_answers, combined_chunks, language)
        print("answer: ", answer, combined_chunks)
        return answer, combined_chunks
        
        if ("无法回答" in answer or 'Unable to answer' in answer):
            return answer, combined_chunks
        
        #4. Fine-tune retriever
        retrieve_answer = get_remove_names_from_text(answer, doc_names)
        # modified_query_text is not defined here, using query_text (defined at start of function)
        modified_query_text = get_remove_names_from_text(query_text, doc_names)
        final_retrieve = modified_query_text + " " + retrieve_answer
        print("[4] rerieve for final answer: {}".format(final_retrieve))
        retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
        
        return_chunks = []
        for index, chunk in enumerate(retrieved_small_chunks):
            return_chunks.append(small_chunks[chunk['chunk_index']])
        print('final chunks: ', len(return_chunks))
        return answer, combined_chunks


def retrieve_single_query(query_text, language="en", prediction=None, doc_id=[], doc_names=[]):
    modified_query_text = get_remove_names_from_text(query_text, doc_names)
    print("query_text: ", query_text)
    print("modified_query_text: ", modified_query_text)

    # 1. Retrieve bigger chunks(use BM25)
    print("[1] retrieve with bigger chunks:")
    retrieved_chunks = []
    retrieved_chunks.extend(retrieve_bigger_chunks(query_text, language, prediction, doc_id, doc_names))

    # 2. Retrieve smaller chunks(use BM25)
    print("[2] retrieve with smaller chunks and extract document name:")
    small_retrieved_chunks, small_chunks = create_smaller_chunks_without_names(language, retrieved_chunks, doc_names)
    retriever_2 = create_retriever(small_retrieved_chunks, language)
    retrieved_small_chunks = retriever_2.retrieve(modified_query_text, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])

    print('chunks: ', len(return_chunks))

    # 3. Generate Answer
    print("[3] generate answer:")
    answer = generate_answer(query_text, return_chunks, language)
    print("answer: ", answer)
    if ("无法回答" in answer or 'Unable to answer' in answer):
        return answer, return_chunks
    
    #4. Fine-tune retriever
    retrieve_answer = get_remove_names_from_text(answer, doc_names)
    final_retrieve = modified_query_text + " " + retrieve_answer
    print("[4] rerieve for final answer: {}".format(final_retrieve))
    retrieved_small_chunks = retriever_2.retrieve(final_retrieve, top1_check=True) # retrieve for higher than the top 1 score * 0.5
    
    return_chunks = []
    for index, chunk in enumerate(retrieved_small_chunks):
        return_chunks.append(small_chunks[chunk['chunk_index']])
    print('final chunks: ', len(return_chunks))
    return answer, return_chunks

def generate_sub_query_answer(query, context, language="en", doc_names=[]): 
    if language == "en":
        prompt = """
      You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
      Please answer with in one concise answer. Do not repeat the question or the context.

      Question:
      {query} 

      Context:
      {context} 
      
      Answer:
    """
    else:
        #zh
        prompt = """
      你是一个问答助手。请使用以下检索上下文回答问题。
      请给出一个简短的回答。不要重复问题或上下文。

      Question:
      {query} 
      Context:
      {context} 
      
      Answer:
    """
    context = "\n".join([chunk['page_content'] for chunk in context])
    prompt = prompt.format(query=query, context=context)
    print(prompt)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
    }, prompt=prompt)

    answer = response["response"]
    print("answer: ", answer)
    return answer


# generate_combined_questions_answer(query_text, queries, combined_answers, combined_chunks, language)
def generate_combined_questions_answer(original_query, queries, combined_answers, combined_chunks, language="en", doc_names=[]): 
    if language == "en":
        prompt = """
You are an assistant for question-answering tasks. 
You are given a question and a list of sub questions with their answers.
Use the sub questions and answers to answer the original question.
Please merge all fragments into a single consistent answer. Keep the answer extremely concise and short.
Also, if you don't know the answer, use the following pieces of retrieved context to answer the question, or answer: "Unable to answer."

Original Question:
{query} 

Sub Question with Answers:
{sub_query}

Combined Context:
{context} 

Answer:
    """
    else:
        #zh
        prompt = """
你是一个问答助手。请根据子问题和答案回答原始问题。
请合并答案，并且简洁清楚成包含主詞的一句话，不要重复子问题。
若不知道答案，请回答：“无法回答”。

原始问题：
{query} 

子问题和答案：
{sub_query}

组合上下文：
{context} 
答案：
    """
    
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"问题: {query[1]}\n答案: {answer}" for query, answer in zip(queries, combined_answers)])
    prompt = prompt.format(query=original_query, context=context, sub_query=sub_query)
    print(prompt)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "max_tokens": 1024,
    }, prompt=prompt)

    answer = response["response"]
    print("answer: ", answer)
    return answer

def construct_multiple_questions(query, language="en", doc_names=[]):
    if language == "en":
        prompt = """
    You are an assistant for question-answering tasks. 
    Here is a question that contains multiple informations:
    {query}

    Please reframe this into multiple independent questions, based on the provided document names.
    Document names: {doc_names}
    
    Output the questions as a Python list of strings.
    Example: [(Document name, 'Question 1?'), (Document name, 'Question 2?')]
    
    Answer: 
    """
    else:
        prompt = """
    You are an assistant for question-answering tasks. 
    Here is a question that contains multiple informations:
    {query}

    Please reframe this into multiple independent questions, based on the provided document names.
    Document names: {doc_names}
    
    Output the questions as a Python list of strings.
    Example: [(Document name, 'Question 1?'), (Document name, 'Question 2?')]
    
    Answer: 
    """

    prompt = prompt.format(query=query, doc_names=doc_names)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
    }, prompt=prompt)

    queries = response["response"]
    print("queries: ", queries)
    try:
        return ast.literal_eval(queries)
    except:
        return []


########## Helper Functions ##########

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