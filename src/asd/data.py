"""Index audio clips. Columns: path, machine, id, split (train/val/test), label."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEED, VAL_FRACTION

COLUMNS = ["path", "machine", "id", "split", "label"]
_DCASE = re.compile(r"(normal|anomaly)_id_(\d+)_\d+\.wav$")


def _hold_out(df, rng):
    """Move a fraction of the training normals of each machine ID to 'val'."""
    for _, g in df[df.split == "train"].groupby("id"):
        n = max(1, round(len(g) * VAL_FRACTION))
        df.loc[rng.choice(g.index, n, replace=False), "split"] = "val"
    return df


def index_dcase(root, machine):
    """root/<machine>/{train,test}/{normal,anomaly}_id_XX_XXXXXXXX.wav"""
    rows = []
    for split in ("train", "test"):
        for p in sorted((Path(root) / machine / split).glob("*.wav")):
            m = _DCASE.search(p.name)
            if m is None:
                raise ValueError(f"unexpected file name: {p}")
            rows.append((str(p), machine, f"id_{m[2]}", split, int(m[1] == "anomaly")))
    if not rows:
        raise FileNotFoundError(f"no DCASE clips under {root}/{machine}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    assert not ((df.split == "train") & (df.label == 1)).any()
    return _hold_out(df, np.random.default_rng(SEED))


def index_mimii(root, machine):
    """root/<machine>/id_XX/{normal,abnormal}/NNNNNNNN.wav for one SNR.

    Test = all abnormal clips + as many random normals. The draw depends only on
    sorted names, so the same clips are in the same split at every SNR.
    """
    rng, rows = np.random.default_rng(SEED), []
    for d in sorted((Path(root) / machine).glob("id_*")):
        normal = sorted((d / "normal").glob("*.wav"))
        abnormal = sorted((d / "abnormal").glob("*.wav"))
        if len(normal) <= len(abnormal):
            raise ValueError(f"{d}: not enough normal clips to train on")
        test = set(rng.choice(len(normal), len(abnormal), replace=False).tolist())
        rows += [(str(p), machine, d.name, "test" if i in test else "train", 0)
                 for i, p in enumerate(normal)]
        rows += [(str(p), machine, d.name, "test", 1) for p in abnormal]
    if not rows:
        raise FileNotFoundError(f"no MIMII clips under {root}/{machine}")
    return _hold_out(pd.DataFrame(rows, columns=COLUMNS), rng)
