from Name_Router.prompt_processor import get_formated_prompt, BasePromptType
from pathlib import Path
from Name_Router.LLMClient import LLMClient

prompts_path = Path(__file__).parent / "prompt.yaml"

class PromptType(BasePromptType):
    # Registered prompt types
    COMPLEX_ANSWER = ("ComplexAnswer", ["query", "context"])
    COMBINE_ANSWER = ("CombineAnswer", ["query", "context", "sub_query"])
    COMPARE_ANSWER = ("CompareAnswer", ["query", "context", "sub_query", "answer"])
    MEDICAL_ANSWER = ("MedicalAnswer", ["query", "context"])
    SIMPLE_ANSWER = ("SimpleAnswer", ["query", "context"])

def generate_complex_answer(query, docs, language="en"):
    context = "\n".join([doc['content'] for doc in docs])
    prompt = get_formated_prompt(prompts_path,PromptType.COMPLEX_ANSWER, language=language, query=query, context=context)
    return LLMClient().generate(prompt, {
        "num_ctx": 32768,
        "temperature": 0.0, 
        "num_predict": 256,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "stop": ['\n\n'],
    })

def generate_combined_questions_answer(original_query, queries, combined_answers, combined_chunks, language="en", doc_names=[]): 
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"Question: {query[1]}\nAnswer: {answer}" for query, answer in zip(queries, combined_answers)])
    prompt = get_formated_prompt(prompts_path, PromptType.COMBINE_ANSWER, language=language, query=original_query, context=context, sub_query=sub_query)
    return LLMClient().generate(prompt, {
        "num_ctx": 8192,
         "temperature": 0.0,
         "top_p": 0.9,
         "top_k": 40,
         "num_predict": 128,
         "stop": ["\n\n"],
    })

def generate_compare_answer(original_query, queries, combined_answers, combined_chunks, compare_result, language="en", doc_names=[]): 
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"Question: {query[1]}\nAnswer: {answer}" for query, answer in zip(queries, combined_answers)])
    final_prompt = get_formated_prompt(prompts_path, PromptType.COMPARE_ANSWER, language=language, answer=compare_result, query=original_query, context=context, sub_query=sub_query)
    return LLMClient().generate(final_prompt, {
         "temperature": 0.0,
         "num_predict": 128,
            "top_p": 0.9,
            "top_k": 40,
         "stop": ["\n\n"],
    })

def generate_medical_answer(query, docs, language):
    context = "\n".join([doc['content'] for doc in docs])
    prompt = get_formated_prompt(prompts_path, PromptType.MEDICAL_ANSWER, language=language, query=query, context=context)
    return LLMClient().generate(prompt, {
        "num_ctx": 32768,
        "temperature": 0.0,
        "num_predict": 128,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["\n\n"],
    })

def generate_simple_answer(query, context, language="en", doc_names=[]): 
    context = "\n".join([chunk['page_content'] for chunk in context])
    prompt = get_formated_prompt(prompts_path, PromptType.SIMPLE_ANSWER, language=language, query=query, context=context)
    return LLMClient().generate(prompt, {
         "temperature": 0.0,
         "num_predict": 64,
         "top_p": 0.9,
         "top_k": 40,
         "frequency_penalty": 0.5,
         "presence_penalty": 0.5,
         "stop": ["\n\n"],
    })