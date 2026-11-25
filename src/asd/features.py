"""Log-mel spectrograms (dB, same settings as the DCASE 2020 baseline), cached on disk."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .config import CACHE, CLIP_SECONDS, HOP, N_FFT, N_FRAMES, N_MELS, SR


@lru_cache
def _mel_basis():
    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def prepare(y, sr):
    """Resample to 16 kHz and pad/crop to 10 s."""
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return librosa.util.fix_length(y, size=SR * CLIP_SECONDS)


def load_audio(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return prepare(y[:, 0], sr)  # first microphone


def log_mel(y):
    power = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    return (10 * np.log10(_mel_basis() @ power + 1e-10)).astype(np.float32)


def cached_log_mels(name, index_fn):
    """Return (index, array [clips, mels, frames]), computing them only once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    csv, npy = CACHE / f"{name}.csv", CACHE / f"{name}_logmel.npy"
    if csv.exists() and npy.exists():
        return pd.read_csv(csv), np.load(npy)
    df = index_fn()
    x = np.empty((len(df), N_MELS, N_FRAMES), np.float32)

    def work(i):
        x[i] = log_mel(load_audio(df.path[i]))

    with ThreadPoolExecutor(6) as ex:
        list(tqdm(ex.map(work, range(len(df))), total=len(df), desc=name, mininterval=20))
    np.save(npy, x)
    df.to_csv(csv, index=False)  # written last: marks the cache complete
    return df, x
