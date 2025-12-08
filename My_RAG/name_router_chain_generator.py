from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from utils import load_ollama_config
from ollama import Client
import ast
from generator import generate_answer
import json


def construct_multiple_questions(query, language="en", doc_names=[]):
    prompt = """
###Instruction:
You are a query breakdown assistant. Your task is to generate a sub-question for each item based on the user's main query.

###Steps:
1. Break down the User Query into multiple sub-questions, each sub-question should be clear and concise.
2. The sub-questions should be as similar as possible to the original query.
3. Use the doc_names to generate the sub-questions.

###CRITICAL OUTPUT RULES:
You must output a valid JSON list of objects.
Do not include markdown formatting (like ```json). Just the raw JSON.

Each object must have exactly two keys: "doc_name" and "sub_question".
doc_names: {doc_names}
User Query: {query}
"""

    prompt = prompt.format(query=query, doc_names=doc_names)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.1, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
    }, prompt=prompt)

    queries = response["response"]
    print("queries: ", queries)
    return queries


def generate_sub_query_answer(query, context, language="en", doc_names=[]): 
    prompt = """
### Role
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
### Task
Please answer with in one concise answer. Do not repeat the question or the context.

### Steps
1. Analyze the Question to understand what specific information is needed.
2. Scan the Reference Data to find the exact match.
3. If the answer is single and involves specific information (e.g., name, date, amount, location, project, event), answer with ONLY the specific requested information within the reference data.
4. If the answer is found, write it down.

Question:
{query} 

Context:
{context} 

### Answer
    """
    context = "\n".join([chunk['page_content'] for chunk in context])
    prompt = prompt.format(query=query, context=context)
    print(prompt)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "max_tokens": 1024,
         "top_p": 0.9,
         "top_k": 40,
         "frequency_penalty": 0.5,
         "presence_penalty": 0.5,
         "stop": ["\n\n"],
    }, prompt=prompt)

    answer = response["response"]
    print("answer: ", answer)
    return answer

def generate_combined_questions_answer(original_query, queries, combined_answers, combined_chunks, language="en", doc_names=[]): 
    if language == "en":
        prompt = """
        ### Role
You are a concise synthesis assistant. 
Your goal is to construct a single, direct answer to the User's Original Question by synthesizing provided Sub-Questions and Context.

### Task
1. **Prioritize Sub-QA:** Your primary source of truth is the "Sub Question with Answers" section.
2. **Support with Context:** Use the "Combined Context" to verify facts.
3. **Synthesize:** Combine the facts into one smooth, coherent response, and make sure it is concise, use the same format in the context.
4. If questions is comparing two things, just compare the two things in the answer. No need to concluding at the end.
5. **Fallback:** If the answer cannot be found in the provided text, strictly output: "Unable to answer."

### Input Data

**Original Question:**
{query} 

**Sub Question with Answers:**
{sub_query}

**Combined Context:**
{context} 

### Final Answer
"""
    else:
        #zh
        prompt = """### Role
You are a concise synthesis assistant. Your goal is to construct a single, direct answer to the User's Original Question by synthesizing provided Sub-Questions and Context.

### Task
1. **Prioritize Sub-QA:** Your primary source of truth is the "Sub Question with Answers" section.
2. **Support with Context:** Use the "Combined Context" to verify facts.
3. **Synthesize:** Combine the facts into one smooth, coherent response, and make sure it is concise, use the same format in the context.
4. If questions is comparing two things, just compare the two things in the answer. No need to concluding at the end.
5. **Fallback:** If the answer cannot be found in the provided text, strictly output: "无法回答"
6. Answer in Simplified Chinese.

### Input Data

**Original Question:**
{query} 

**Sub Question with Answers:**
{sub_query}

**Combined Context:**
{context} 

### Final Answer
    """
    
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"Question: {query[1]}\nAnswer: {answer}" for query, answer in zip(queries, combined_answers)])
    prompt = prompt.format(query=original_query, context=context, sub_query=sub_query)
    print(prompt)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "max_tokens": 1024,
         "stop": ["\n\n"],
    }, prompt=prompt)

    answer = response["response"]
    print("answer: ", answer)
    return answer
