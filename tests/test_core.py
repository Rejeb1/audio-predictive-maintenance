import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.autoencoder import AEDetector, patch_starts
from asd.data import index_dcase, index_mimii
from asd.features import log_mel, prepare
from asd.metrics import decisions
from asd.models import ConvAE, DenseAE


def test_log_mel_shape():
    y = prepare(np.random.randn(44100 * 3).astype(np.float32), 44100)
    assert log_mel(y).shape == (C.N_MELS, C.N_FRAMES)


def test_model_shapes():
    assert DenseAE()(torch.randn(4, 640)).shape == (4, 640)
    assert ConvAE()(torch.randn(2, 1, 128, 64)).shape == (2, 1, 128, 64)


def test_patches_cover_clip():
    s = patch_starts(C.N_FRAMES)
    assert s[0] == 0 and s[-1] + C.PATCH == C.N_FRAMES


def test_fit_reduces_error_and_roundtrips(tmp_path):
    x = np.random.randn(6, C.N_MELS, C.N_FRAMES).astype(np.float32)
    for kind in ("dense", "conv"):
        d = AEDetector(kind).fit(x, x[:2], epochs=2, log=lambda r: None)
        assert d.history[1]["train"] < d.history[0]["train"]
        d.save(tmp_path / "m.pt")
        assert np.allclose(AEDetector.load(tmp_path / "m.pt").score(x), d.score(x), rtol=1e-4)
        assert d.error_map(x[0]).shape == x[0].shape


def test_dcase_split_has_no_anomaly_or_leak(tmp_path):
    for split, lab in (("train", "normal"), ("test", "normal"), ("test", "anomaly")):
        d = tmp_path / "pump" / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            (d / f"{lab}_id_00_{i:08d}.wav").touch()
    df = index_dcase(tmp_path, "pump")
    assert df[df.split != "test"].label.sum() == 0
    assert (df.split == "val").sum() == 2 and df.path.is_unique


def test_mimii_split_same_at_each_snr(tmp_path):
    got = []
    for snr in "ab":
        for lab, n in (("normal", 30), ("abnormal", 5)):
            d = tmp_path / snr / "fan" / "id_00" / lab
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i:08d}.wav").touch()
        got.append(index_mimii(tmp_path / snr, "fan"))
    assert (got[0].split == got[1].split).all()
    assert got[0][got[0].split != "test"].label.sum() == 0


def test_threshold_ignores_test_scores():
    base = dict(model="m", machine="pump", id="id_00")
    df = pd.DataFrame([dict(base, split="val", label=0, score=float(i)) for i in range(11)]
                      + [dict(base, split="test", label=1, score=1e6)])
    assert decisions(df, 90).threshold.iloc[0] == 9.0


def test_guard_rejects_hardware_style_spike():
    import pytest
    from asd.autoencoder import TrainingFault, check_epoch
    norms = torch.full((300,), 0.12)
    check_epoch(5, norms, 0.43, 0.43)  # healthy epoch passes
    loud = norms.clone()
    loud[7] = 0.12 * 145  # observed on a Kaggle T4, not a fault by itself
    check_epoch(5, loud, 0.43, 0.43)
    for bad in (5.1e8, 1.1e7, float("nan"), float("inf")):
        n = norms.clone()
        n[7] = bad
        with pytest.raises(TrainingFault):
            check_epoch(5, n, 0.43, 0.43)
    with pytest.raises(TrainingFault):
        check_epoch(5, norms, 0.95, 0.43)
