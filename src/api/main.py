from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import json
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Resample

from src.data.dataloader import ASRDataset, collate_fn
from src.models.model import ASRLightningConformer
from .config import ServiceConfig

app = FastAPI()

config: ServiceConfig | None = None
model: ASRLightningConformer | None = None
idx_to_vocab: dict[int, str] = {}

mel_transform = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
amp_to_db = AmplitudeToDB()


def load_model(cfg_path: str) -> None:
    global config, model, idx_to_vocab
    config = ServiceConfig(cfg_path)
    model = ASRLightningConformer.load_from_checkpoint(str(config.model_checkpoint))
    model.eval()
    model.freeze()
    with open(config.vocabulary_json, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_vocab = {i + 1: ch for i, ch in enumerate(vocab)}


def preprocess_audio(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    mel_spec = mel_transform(waveform)
    mel_spec = amp_to_db(mel_spec)
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
    return mel_spec.squeeze(0).transpose(0, 1)


def decode(tokens: List[int]) -> str:
    return "".join(idx_to_vocab.get(t, "") for t in tokens)


def levenshtein_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


@app.on_event("startup")
def startup_event() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "service_config.yaml"
    load_model(str(cfg_path))


@app.post("/download_data")
async def download_data(
    kaggle_username: str,
    kaggle_key: str,
    competition: str = "asr-numbers-recognition-in-russian",
    output_dir: str = "data",
) -> dict:
    from src.data import data_download

    data_download.main.callback(
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
        competition=competition,
        output_dir=output_dir,
    )
    return {"status": "downloaded"}


@app.post("/prepare_data")
async def prepare_data(
    numbers_data_path: str = "data/numbers",
    words_data_path: str = "data/words",
    tokens_data_path: str = "data/tokens",
) -> dict:
    from src.data import data_prepare

    data_prepare.main.callback(
        numbers_data_path=numbers_data_path,
        words_data_path=words_data_path,
        tokens_data_path=tokens_data_path,
    )
    return {"status": "prepared"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        return {"error": "model not loaded"}
    audio_bytes = await file.read()
    waveform, sr = torchaudio.load(BytesIO(audio_bytes))
    spec = preprocess_audio(waveform, sr)
    length = torch.tensor([spec.shape[0]], dtype=torch.long)
    spec = spec.unsqueeze(0)
    with torch.no_grad():
        logits, out_lens = model(spec, length)
        eff = out_lens // model.hparams.subsampling_factor
        pred = torch.argmax(logits, dim=-1)[0, : eff[0]]
    text = decode(pred.tolist())
    return {"transcription": text}


@app.get("/metrics")
def metrics() -> dict:
    if model is None or config is None:
        return {"error": "model not loaded"}
    if not config.val_csv or not config.val_audio_dir:
        return {"error": "validation data not configured"}
    ds = ASRDataset(
        str(config.val_csv),
        str(config.val_audio_dir),
        str(config.vocabulary_json),
        target_sample_rate=16000,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    total_dist = 0
    total_len = 0
    for specs, lens, targets, t_lens in loader:
        with torch.no_grad():
            logits, out_lens = model(specs, lens)
            eff = out_lens // model.hparams.subsampling_factor
            pred = torch.argmax(logits, dim=-1)[0, : eff[0]]
        target_seq = targets[: t_lens[0]]
        pred_text = decode(pred.tolist())
        target_text = decode(target_seq.tolist())
        total_dist += levenshtein_distance(target_text, pred_text)
        total_len += len(target_text)
    cer = total_dist / max(1, total_len)
    return {"cer": cer}
