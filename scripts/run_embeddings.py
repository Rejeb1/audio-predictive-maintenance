"""python scripts/run_embeddings.py --dataset dcase --machine pump

Settings (k=5, PCA 32, 2 components) are fixed in advance, not tuned on test anomalies.
'type' = one detector per machine type, like the autoencoders (fair comparison).
'id'   = one detector per machine ID (ID is known in DCASE 2020 and in a real plant).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_ae import C, detection_table, load
from asd.embeddings import GMMDetector, KNNDetector, cached_embeddings

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dcase")
    ap.add_argument("--machine", required=True)
    a = ap.parse_args()
    df, _ = load(a.dataset, a.machine)
    e = cached_embeddings(f"{a.dataset}_{a.machine}", df)
    frames = []
    for scope in ("type", "id"):
        groups = [df.index] if scope == "type" else [g.index for _, g in df.groupby("id")]
        for label, make in (("knn", KNNDetector), ("gmm", GMMDetector)):
            for idx in groups:
                g = df.loc[idx]
                det = make().fit(e[g.index[g.split == "train"]])
                rest = g[g.split != "train"]
                frames.append(rest.assign(model=f"panns_{label}_{scope}", dataset=a.dataset,
                                          score=det.score(e[rest.index])))
    out = pd.concat(frames, ignore_index=True)
    out["path"] = [Path(p).name for p in out.path]
    out_dir = C.RESULTS / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{a.dataset}_{a.machine}_panns.csv", index=False)
    assert np.isfinite(out.score).all()
    print(detection_table(out).round(4).to_string())
