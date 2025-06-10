from pathlib import Path
from typing import Optional

import yaml


class ServiceConfig:
    """Configuration for the API service."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.model_checkpoint: Path = Path(data.get("model_checkpoint", "model.ckpt"))
        self.vocabulary_json: Path = Path(data.get("vocabulary_json", "vocabulary.json"))
        self.val_csv: Optional[Path] = Path(data["val_csv"]) if data.get("val_csv") else None
        self.val_audio_dir: Optional[Path] = Path(data["val_audio_dir"]) if data.get("val_audio_dir") else None
