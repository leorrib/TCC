import yaml
from pathlib import Path


class ConfigLoader:
    """Simple utility to load YAML configuration files."""

    @staticmethod
    def load(config_name: str = "base.yaml") -> dict:
        config_path = Path(__file__).resolve().parents[1] / "config" / config_name
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
