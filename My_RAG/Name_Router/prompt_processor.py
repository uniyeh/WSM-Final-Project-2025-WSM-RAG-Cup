from enum import Enum
from pathlib import Path
import yaml

# Here for handle prompts loading and formatting
# - PromptType: Enum defining supported prompts and their required keys.
# - get_formated_prompt: Validates and fills prompt templates from prompts.yaml.

class BasePromptType(Enum):
    def __init__(self, key, required_keys):
        self.key = key
        self.required_keys = required_keys

def get_formated_prompt(prompts_path: Path, prompt_type: BasePromptType, language="en", **kwargs):
    if not isinstance(prompt_type, Enum):
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    # 1. Validation: Ensure all required keys are present
    missing = [k for k in prompt_type.required_keys if k not in kwargs]
    if missing:
        raise ValueError(f"Missing keys for {prompt_type.name}: {missing}")

    # 2. Load
    prompt_data = load_prompts(prompts_path, prompt_type.key)
    
    # 3. If prompt is language-specific, check language
    if isinstance(prompt_data, dict):
        if language not in prompt_data:
             raise ValueError(f"Language {language} not supported for prompt {prompt_type.name}")
        return prompt_data[language].format(**kwargs)
    
    return prompt_data.format(**kwargs)

def load_prompts(prompts_path: Path, type: str = "default"):
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found at: {path}")
    
    with open(path, "r") as file:
        data = yaml.safe_load(file)
        if type not in data:
            raise KeyError(f"Key '{type}' not found in prompts file.")
        return data[type]