from ollama import Client
from pathlib import Path
import yaml
from utils import load_ollama_config


def load_prompts(type="default"):
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        raise FileNotFoundError("Prompts file not found.")
    with open(prompts_path, "r") as file:
        return yaml.safe_load(file)[type]


def generate_answer(query, context_chunks, language="en", type="default"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompts = load_prompts(type)
    if language not in prompts:
        print(f"Warning: Language '{language}' not found in prompts. Falling back to 'en'.")
        language = "en"
        
    prompt_template = prompts[language]
    prompt = prompt_template.format(query=query, context=context)
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], options={
        #  "num_ctx": 8192, # [4096, 8192, 32768]
         "temperature": 0.5, # [0.0, 1.0], 0.0 is more deterministic, 1.0 is more random and creative
         "top_p": 0.9,
         "top_k": 40,
         "max_tokens": 2048,
         "frequency_penalty": 0.0,
         "presence_penalty": 0.0,
         "stop": ["\n\n"],
    }, prompt=prompt)
    # print("DEBUG: Full Ollama Response:", response)
    return response["response"]


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)