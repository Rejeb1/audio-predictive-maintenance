"""Demo inference: dense autoencoder in ONNX Runtime (CPU, no torch)."""
import librosa
import numpy as np
import soundfile as sf

from .config import CLIP_SECONDS, CONTEXT, N_MELS, SR
from .features import log_mel  # noqa: F401  (re-exported for app.py)

BAND = 16  # mel bands per displayed frequency region


class AudioError(ValueError):
    """The clip cannot be analysed; the message is shown to the user."""


def read_clip(source):
    """First channel, 16 kHz, at most 10 s. Not padded: silence would look abnormal."""
    try:
        with sf.SoundFile(source) as f:
            if f.channels > 16 or not 1000 <= f.samplerate <= 400_000:
                raise AudioError("Unsupported audio format.")
            y = f.read(f.samplerate * CLIP_SECONDS, dtype="float32", always_2d=True)[:, 0]
            sr = f.samplerate
    except AudioError:
        raise
    except Exception as e:
        raise AudioError("Unreadable audio file.") from e
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < SR:
        raise AudioError("Clip too short: at least 1 second is needed.")
    if not np.isfinite(y).all() or np.abs(y).max() < 1e-5:
        raise AudioError("The clip is silent or invalid.")
    return y


def analyse(session, x):
    """log-mel [mels, T] -> (score, error map [mels, T])."""
    w = np.lib.stride_tricks.sliding_window_view(x, CONTEXT, axis=1).transpose(1, 2, 0)  # [P, 5, mels]
    flat = w.reshape(len(w), -1).copy()
    rec = session.run(None, {"x": flat})[0]
    sq = (rec - flat) ** 2
    err, cnt = np.zeros_like(x), np.zeros(x.shape[1])
    for k in range(CONTEXT):  # average the overlapping windows
        s = slice(k, k + len(w))
        err[:, s] += sq.reshape(len(w), CONTEXT, N_MELS)[:, k].T
        cnt[s] += 1
    return float(sq.mean()), err / cnt


def band_edges_hz():
    f = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR / 2)
    cut = (f[:-1] + f[1:]) / 2  # halfway between neighbouring band centres
    cut[0], cut[-1] = 0, SR / 2
    return [(float(cut[b]), float(cut[b + BAND])) for b in range(0, N_MELS, BAND)]


def band_means(v):
    """[mels, ...] -> [8, ...]"""
    return np.stack([v[b:b + BAND].mean(0) for b in range(0, N_MELS, BAND)])
