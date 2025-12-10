from chunker import chunk_documents
from runtime_chunker import chunk_row_chunks
from retriever import create_retriever, get_chunks_from_db
from rank_bm25 import BM25Okapi
from utils import load_ollama_config
from ollama import Client
import ast
from generator import generate_answer
import json

def query_classifier(query, language="en"):
    if (language == 'en'):
        prompt = """
### Role
You are a Query Classifier.

### Task
Analyze the user's query and classify it into one of two categories:
1. "SIMPLE": A direct factual question looking for a single specific attribute, definition, or fact about a single entity.
2. "COMPLEX": A question that requires comparing two or more entities, aggregating data (counting, summing), or reasoning across multiple timeframes (multi-hop).

### Output
Return ONLY the category name: "SIMPLE" or "COMPLEX".

### Examples
Query: "When did Green Fields Agriculture Ltd. appoint a new CEO?" -> SIMPLE
Query: "What percentage of equity did Green Fields Agriculture Co. acquire from Fertile Land Farms in September 2018?" -> SIMPLE
Query: "According to the hospitalization records of Bridgewater General Hospital, what is the preliminary diagnosis for J. Reyes?" -> SIMPLE
Query: "According to the judgment of Charleston, Linden, Court, what is the total amount of damage caused by J. Gonzalez in all her crimes?" -> COMPLEX
Query: "How did compliance and regulatory updates in May 2019 contribute to reduced legal risk by December 2019?" -> COMPLEX
Query: "List all events for Company A and Company B." -> COMPLEX

### User Query
{query}
    """
    else:
        prompt = """
### Role
You are a Query Intent Classifier.

### Task
Analyze the "User Query" and predict the complexity of the answer required.
Classify the query into one of two categories: **SIMPLE** or **COMPLEX**.

### Classification Rules

**1. SIMPLE (Direct Lookup)**
* **Definition:** The answer is explicitly written in the text as a single attribute. You do not need to calculate anything.
* **Includes:** Names, Dates, Locations, Prices, ID numbers, or a pre-calculated total (e.g., "What is the Total Revenue?").
* **Keywords:** "When", "Who", "What is the ID", "What amount".

**2. COMPLEX (Calculation & Aggregation)**
* **Definition:** The answer requires **Counting** items, **Summing** values, or **Listing** multiple things. If the model has to count "1, 2, 3..." to get the answer, it is COMPLEX.
* **Includes:** Counting frequency, Listing items, Comparing A vs B, Explaining "How".
* **Keywords:** "How many items" (几项), "Count" (统计), "List all" (列出), "Difference" (区别), "How" (如何).

### Critical Distinction: "Numbers"
* "What is the price?" -> **SIMPLE** (The number exists in the text).
* "How many products were sold?" -> **COMPLEX** (You must count the rows/items).

### Examples for Training
* "What is the CEO's name?" -> **SIMPLE** (Lookup Name)
* "What is the total cost listed on the invoice?" -> **SIMPLE** (Lookup pre-existing number)
* "How many times did the CEO visit the factory?" -> **COMPLEX** (Counting required)
* "List all board members." -> **COMPLEX** (Listing required)
* "进行了几项？" -> **COMPLEX** (Requires counting the exam items)
* "进行了几个？" -> **COMPLEX** (Requires counting the exam items)

### Input Data
User Query: {query}
### Output
Reasoning: [One sentence explaining why]
Label: [SIMPLE or COMPLEX]
    """
    prompt = prompt.format(query=query)
    print("query_classifier: ", prompt)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.1, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "top_p": 0.9,
         "top_k": 40,
         "max_tokens": 2048,
        #  "stream": True,
    }, prompt=prompt)
    print("query_classifier: ", response["response"])
    return response["response"]

def generate_complex_answer(query, docs, language="en"):
    if language == "en":
        prompt="""
### Role
You are a precise Document Summarizer.

### Task
Answer the user's question by synthesizing the provided court judgments. 
Your output must be a single, fluent narrative sentence that cites the specific court and provides the exact supporting informations.
Only output the answer, do not include any additional text.
If you are not able to answer the question, output "Unable to answer.".

### Reference Data
<context>
{context}
</context>

### Question
{query}

### Output Style Rules (CRITICAL)
1. **Source Citation:** Start every claim with "According to the ...".
2. **Handling Conflicts:** - If two different courts provide different facts for the same person, use a contrast structure.
3. **Handling Counts & Frequencies:**
   - You must state the total number AND list the specific timestamps/dates.

### Examples of Desired Output
**User Query:** "How much did V. Martin embezzle?"
**Output:** "According to the judgment of Bayside, Roseville, Court, the total amount embezzled by V. Martin is $700,000. However, according to the ruling in Vandalia, Bayside Court, no money was embezzled by V. Martin."

### Final Answer
"""
    else:
        prompt='''
### Role
You are a precise Document Summarizer.

### Task
Answer the user's question by synthesizing the provided court judgments. 
Your output must be a single, fluent narrative sentence that cites the specific court and provides the exact supporting informations.
Only output the answer, do not include any additional text.
If you are not able to answer the question, output "无法回答".
Answer in Simplified Chinese.

### Reference Data
<context>
{context}
</context>

### Question
{query}

### Output Style Rules (CRITICAL)
1. **Source Citation:** Start every claim with "根据...".
2. **Handling Conflicts:** - If two different courts provide different facts for the same person, use a contrast structure.
3. **Handling Counts & Frequencies:**
   - You must state the total number AND list the specific timestamps/dates.

### Examples of Desired Output
**User Query:** "根据紫陌市人民医院的住院病历，贝某某的初步诊断是什么？"
**Output:** "根据病历，贝某某的初步诊断是右眼急性闭角型青光眼。"

**User Query:** "根据彩虹市桐城区人民法院的判决书，费某一共有几次交通肇事行为？"
**Output:** "根据判决书，费某一共有三次交通肇事行为：2023年2月15日晚上8时至9时、2023年2月28日早上7时至7时30分、2023年3月10日下午2时至2时30分。"

### Final Answer
'''
    prompt = prompt.format(context="\n".join([doc['content'] for doc in docs]), query=query)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
        "num_ctx": 32768,
        "temperature": 0.3, 
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stop": ['\n\n'],
    }, prompt=prompt)
    print("\ngenerate_complex_answer: ", response)
    return response["response"]

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

Each object must have exactly two keys: "doc_name" and "sub_question", "doc_name" should be from doc_names.
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

def compare_then_generate_answer(original_query, queries, combined_answers, combined_chunks, language="en", doc_names=[]): 
    if language == "en":
        prompt = """
### Role
You are a precise data comparison assistant.

### Task
1. **Prioritize Sub-QA:** You must compare the entities based on the question. Your primary source of truth is the "Sub Question with Answers" section.
2. **Support with Context:** Use the "Combined Context" to verify facts.

**Question:**
{query} 

**Sub Question with Answers:**
{sub_query}

### Context
{context}

### Output Rules
1. Identify the two entities being compared.
2. Extract the specific values for the attribute in question (e.g., time, cost, score).
3. Determine the "winner" based on the user's criteria (e.g., who is earlier? who is cheaper?).
4. Output in a comparison_sentence:
"<Winner Name> + <Attribute Context> + <Comparative Adjective>"

### Answer
"""
    else:
        #zh
        prompt = """
### Role
You are a precise data comparison assistant.

### Task
You will receive a 
You must compare the entities based on the question.

**Original Question:**
{query} 


**Sub Question with Answers:**
{sub_query}

### Context
{context}


### Output Rules
1. Identify the two entities being compared.
2. Extract the specific values for the attribute in question (e.g., time, cost, score).
3. Determine the "winner" based on the user's criteria (e.g., who is earlier? who is cheaper?).
4. Output in a comparison_sentence:
"<Winner Name> + <Attribute Context> + <Comparative Adjective>"
5. Answer in Simplified Chinese.

### Answer
""" 
    
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"Question: {query[1]}\nAnswer: {answer}" for query, answer in zip(queries, combined_answers)])
    prompt = prompt.format(query=original_query, context=context, sub_query=sub_query)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.1, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "max_tokens": 1024,
         "stop": ["\n\n"],
    }, prompt=prompt)

    answer = response["response"]
    print("compare answer: ", answer)

    if language == "en":
        final_prompt="""
### Instruction
You are a Natural Language Generator. 
You will receive a "Raw Context" string containing key phrases separated by symbols (like "+" or "|").
Your task is to convert these fragments into a single, grammatically correct, and fluent sentence.

### Guidelines
1. **Connect Logic:** Add necessary prepositions (of, in, by, to) and possession markers ('s) to make the sentence flow naturally.
2. **Preserve Meaning:** Do not change the original meaning or facts, using all the information in the context.
3. **No Fluff:** Do not add introductory phrases like "The sentence is..." or "Here is the result." Just output the sentence.

### Raw Context
{answer}

### Natural Sentence
"""
    else:
        final_prompt="""
### Instruction
You are a Natural Language Generator. 
You will receive a "Raw Context" string containing key phrases separated by symbols (like "+" or "|").
Your task is to convert these fragments into a single, grammatically correct, and fluent sentence.

### Guidelines
1. **Connect Logic:** Add necessary prepositions (of, in, by, to) and possession markers ('s) to make the sentence flow naturally.
2. **Preserve Meaning:** Do not change the original meaning or facts, using all the information in the context.
3. **No Fluff:** Do not add introductory phrases like "The sentence is..." or "Here is the result." Just output the sentence.
4. Answer in **Simplified Chinese**.

### Raw Context
{answer}

### Natural Sentence
"""
    prompt = final_prompt.format(answer=answer)
    
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
         "temperature": 0.3, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "max_tokens": 1024,
         "stop": ["\n\n"],
    }, prompt=prompt)

    answer = response["response"]
    print("final compare answer: ", answer)

    return answer
