import json
from Name_Router.LLMClient import LLMClient
from Name_Router.prompt_processor import get_formated_prompt, BasePromptType
from pathlib import Path

prompts_path = Path(__file__).parent / "reasoner_prompt.yaml"

class PromptType(BasePromptType):
    INTENT_CLASSIFIER = ("IntentClassifier", ["query"])
    CONSTRUCT_QUESTION = ("ConstructQuestion", ["query", "doc_names"])
    SUB_QUESTION_ANSWER = ("SubQuestionAnswer", ["query", "context"])
    COMPARE_SUB_QUESTIONS_ANSWER = ("CompareSubQuestionsAnswer", ["query", "context", "sub_query"])
    QA_CLASSIFIER = ("QAClassifier", ["query", "answer"])

def query_classifier(query, language="en"):
    prompt = get_formated_prompt(prompts_path, PromptType.INTENT_CLASSIFIER, language=language, query=query)
    return LLMClient().generate(prompt, {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 256,
    })

def construct_multiple_questions(query, language="en", doc_names=[]):
    prompt = get_formated_prompt(prompts_path, PromptType.CONSTRUCT_QUESTION, language=language, query=query, doc_names=doc_names)
    raw_response = LLMClient().generate(prompt, {
        "temperature": 0.0,
    })
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("[Generator] The LLM output was not valid JSON.")
        return []

def generate_sub_query_answer(query, context, language="en", doc_names=[]): 
    context = "\n".join([chunk['page_content'] for chunk in context])
    prompt = get_formated_prompt(prompts_path, PromptType.SUB_QUESTION_ANSWER, language=language, query=query, context=context)
    return LLMClient().generate(prompt, {
        "temperature": 0.0,
        "num_predict": 256,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "stop": ["\n\n"],
    })

def compare_sub_questions_answer(original_query, queries, combined_answers, combined_chunks, language="en", doc_names=[]): 
    context = "\n\n".join([chunk['metadata']['name'] + ": " + chunk['page_content'] for chunk in combined_chunks])
    sub_query = "\n\n".join([f"Question: {query[1]}\nAnswer: {answer}" for query, answer in zip(queries, combined_answers)])
    prompt = get_formated_prompt(prompts_path, PromptType.COMPARE_SUB_QUESTIONS_ANSWER, language=language, query=original_query, context=context, sub_query=sub_query)
    return LLMClient().generate(prompt, {
        "temperature": 0.0,
        "num_predict": 128,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["\n\n"],
    })

def question_and_answer_classifier(query, answer, language="en", doc_names=[]):
    prompt = get_formated_prompt(prompts_path, PromptType.QA_CLASSIFIER, language=language, query=query, answer=answer)
    return LLMClient().generate(prompt, {
        "temperature": 0.0,
        "num_predict": 128,
        "top_p": 0.9,
        "top_k": 40,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "stop": ["\n\n"],
    })

def fallback_to_simple_check(query, answer, language="en", doc_names=[]):
    response = question_and_answer_classifier(query=query, answer=answer, language=language)
    if "SIMPLE" in response.upper():
        return True
    return False
