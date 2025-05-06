import ast
import json
import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    Resample,
)


class ASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, vocabulary_json, target_sample_rate=16000):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        with open(vocabulary_json, "r") as vocabulary_file:
            self._vocab = json.load(vocabulary_file)

        self.mel_transform = MelSpectrogram(sample_rate=target_sample_rate, n_fft=400, hop_length=160, n_mels=80)
        self.amplitude_to_db = AmplitudeToDB()

        self.vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(self._vocab)}
        self.idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(self._vocab)}
        self.num_classes = len(self._vocab) + 1  # +1 for blank token

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])

        waveform, sample_rate = torchaudio.load(file_path)
        mel_spec = self._preprocess_audio(waveform, sample_rate)
        target = self._preprocess_target(str(row["transcription"]))
        return mel_spec, target

    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        return mel_spec

    def _preprocess_target(self, target: str) -> torch.Tensor:
        if target.startswith("[") and target.endswith("]"):
            target = ast.literal_eval(target)
        encoded = [self.vocab_to_idx[c] for c in target if c in self.vocab_to_idx]
        return torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch):
    specs, targets = zip(*batch)
    specs = [s.squeeze(0).transpose(0, 1) for s in specs]
    spec_lengths = [s.shape[0] for s in specs]

    max_len = max(spec_lengths)
    B = len(specs)
    n_mels = specs[0].shape[1]
    padded = torch.zeros(B, max_len, n_mels)
    for i, s in enumerate(specs):
        padded[i, : spec_lengths[i], :] = s

    target_lengths = [t.numel() for t in targets]
    targets_concat = torch.cat(targets)

    return (
        padded,  # [B, T_max, n_mels]
        torch.tensor(spec_lengths, dtype=torch.long),  # [B]
        targets_concat,  # [sum(target_lengths)]
        torch.tensor(target_lengths, dtype=torch.long),  # [B]
    )
