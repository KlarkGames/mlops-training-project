import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

API_URL = "http://localhost:8000/predict"
CFG_PATH = Path("service_config.yaml")


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text())
    df = pd.read_csv(cfg["val_csv"])
    audio_dir = Path(cfg["val_audio_dir"])
    start = time.time()
    responses = []
    for _, row in df.iterrows():
        with open(audio_dir / row["filename"], "rb") as f:
            resp = requests.post(API_URL, files={"file": f})
        responses.append(resp.json())
    duration = time.time() - start
    Path("results.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
    print(f"Sent {len(df)} requests in {duration:.2f}s")


if __name__ == "__main__":
    main()
