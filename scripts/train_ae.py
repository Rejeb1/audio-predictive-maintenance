"""python scripts/train_ae.py --dataset dcase --machine pump --kind dense

dataset: 'dcase', or 'mimii_-6dB' / 'mimii_0dB' / 'mimii_6dB' for the noise study.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.autoencoder import AEDetector
from asd.data import index_dcase, index_mimii
from asd.features import cached_log_mels
from asd.metrics import detection_table


def device_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def load(dataset, machine):
    fn = index_dcase if dataset == "dcase" else index_mimii
    return cached_log_mels(f"{dataset}_{machine}", lambda: fn(C.DATA / dataset, machine))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dcase")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--kind", choices=["dense", "conv"], required=True)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    df, x = load(a.dataset, a.machine)
    print(df.groupby(["id", "split", "label"]).size().to_string(), flush=True)
    t0 = time.time()
    det = AEDetector(a.kind).fit(x[(df.split == "train").values], x[(df.split == "val").values],
                                 a.epochs, log=lambda r: print(r, flush=True))
    print(f"trained in {time.time() - t0:.0f}s")
    name = f"{a.dataset}_{a.machine}_{a.kind}"
    out_dir = C.RESULTS / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    det.save(C.MODELS / f"{name}.pt")
    pd.DataFrame(det.history).assign(device=device_name()).to_csv(
        out_dir / f"{name}_history.csv", index=False)
    keep = (df.split != "train").values
    out = df[keep].assign(model=f"{a.kind}_ae", score=det.score(x[keep]), dataset=a.dataset)
    out["path"] = [Path(p).name for p in out.path]
    out.to_csv(out_dir / f"{name}.csv", index=False)  # last: marks the run complete
    print(detection_table(out).round(4).to_string())
