from ollama import Client
from pathlib import Path
import yaml

class LLMClient:
    def __init__(self):
        self.ollama_config = self.load_ollama_config()
        self.client = Client(host=self.ollama_config["host"])
    
    def generate(self, prompt, options):
        response = self.client.generate(model=self.ollama_config["model"], options=options, prompt=prompt)
        return response["response"]
    
    def load_ollama_config(self) -> dict:
        configs_folder = Path(__file__).parent.parent.parent / "configs"
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