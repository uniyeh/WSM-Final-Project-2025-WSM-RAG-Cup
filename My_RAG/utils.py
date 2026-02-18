import jsonlines
from pathlib import Path
import yaml
import sys
import os
# Add parent directory to path to allow importing from db
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.Connection import Connection
DB_PATH = "db/dataset.db"


def load_jsonl(file_path):
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def save_jsonl(file_path, data):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)

def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]

def load_prompts(file_path: str, type: str = "default"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found at: {path}")
    
    with open(path, "r") as file:
        data = yaml.safe_load(file)
        if type not in data:
            raise KeyError(f"Key '{type}' not found in prompts file.")
        return data[type]

class DatasetConnection(Connection):
    def __init__(self):
        super().__init__(DB_PATH)
        