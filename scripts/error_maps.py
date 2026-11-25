"""Error maps on DCASE clips (CPU), after checking that the models reproduce the Kaggle scores.

Clips are picked by rule: machine ID with the best dense AUC; then the median-score normal,
the highest-score fault and the lowest-score fault (a missed one).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.autoencoder import DEVICE, AEDetector
from asd.features import cached_log_mels

EN = "--en" in sys.argv  # English labels for the slides
SUFFIX = "_en" if EN else ""


def T(fr_text, en_text):
    return en_text if EN else fr_text


assert DEVICE == "cpu", "run with CUDA_VISIBLE_DEVICES=-1 (the local GPU is unreliable)"
(C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
per_id = pd.read_csv(C.RESULTS / "detection_per_id.csv")
for m, fr in (("pump", "Pompe"), ("fan", "Ventilateur")):
    df, x = cached_log_mels(f"dcase_{m}", None)
    df["name"] = [Path(p).name for p in df.path]
    df["pos"] = range(len(df))
    dets = {k: AEDetector.load(C.MODELS / f"dcase_{m}_{k}.pt") for k in ("dense", "conv")}
    for k, det in dets.items():
        s = pd.read_csv(C.RESULTS / "scores" / f"dcase_{m}_{k}.csv").merge(
            df[["name", "split", "pos"]], left_on=["path", "split"], right_on=["name", "split"],
            validate="1:1")
        sub = s.sample(40, random_state=0)
        diff = np.abs(det.score(x[sub.pos.values]) / sub.score.values - 1).max()
        print(m, k, f"max relative score difference local CPU / Kaggle: {diff:.1e}", flush=True)
        assert diff < 1e-2
        if k == "dense":
            sc = s[s.split == "test"]
    q = per_id[(per_id.dataset == "dcase") & (per_id.machine == m) & (per_id.model == "dense_ae")
               & (per_id.id != "mean")]
    best = q.sort_values("auc").id.iloc[-1]
    sc = sc[sc.id == best].sort_values("score")
    nor, ano = sc[sc.label == 0], sc[sc.label == 1]
    picks = [(T("normal (score médian)", "normal (median score)"), nor.iloc[len(nor) // 2]),
             (T("panne, score le plus haut", "fault, highest score"), ano.iloc[-1]),
             (T("panne, score le plus bas", "fault, lowest score"), ano.iloc[0])]
    print(m, best, [(t, r.path, round(r.score, 2)) for t, r in picks])
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True, layout="constrained")
    rows = [("log-mel (dB)", [x[r.pos] for _, r in picks], "viridis")]
    rows += [(T("erreur AE ", "error ") + n, [dets[k].error_map(x[r.pos]) for _, r in picks], "Reds")
             for k, n in (("dense", "dense (dB²)"), ("conv", T("convolutif (normalisée)", "conv (normalised)")))]
    for i, (lab, imgs, cmap) in enumerate(rows):
        lo, hi = np.percentile(np.stack(imgs), [1, 99.5])
        for j, im in enumerate(imgs):
            h = axes[i, j].imshow(im, origin="lower", aspect="auto", cmap=cmap,
                                  vmin=lo if i == 0 else 0, vmax=hi, extent=(0, 10, 0, 128))
        axes[i, 0].set_ylabel(lab + T("\nbande mel", "\nmel band"))
        fig.colorbar(h, ax=axes[i].tolist(), pad=.01)
    for j, (t, r) in enumerate(picks):
        axes[0, j].set_title(f"{t}\n{T('score AE dense', 'dense AE score')} {r.score:.1f}", fontsize=9)
        axes[2, j].set_xlabel(T("temps (s)", "time (s)"))
    fig.suptitle(T(f"DCASE {fr} {best} : même échelle par ligne", f"DCASE {m} {best}: same scale within each row"), fontsize=10)
    fig.savefig(C.RESULTS / "figures" / f"error_map_{m}{SUFFIX}.png", dpi=150)
