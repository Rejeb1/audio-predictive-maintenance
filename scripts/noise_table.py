"""Mean AUC/pAUC per condition, with a stratified bootstrap 95% interval on AUC."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.metrics import detection_table


def mean_auc(groups, rng=None):
    out = []
    for n, a in groups:
        if rng is not None:
            n, a = rng.choice(n, len(n)), rng.choice(a, len(a))
        s = np.sort(n)
        out.append(np.mean((np.searchsorted(s, a, "left") + np.searchsorted(s, a, "right")) / 2) / len(s))
    return np.mean(out)


if __name__ == "__main__":
    files = [p for p in sorted((C.RESULTS / "scores").glob("*.csv")) if not p.stem.endswith("_history")]
    rows = []
    for p in files:
        df = pd.read_csv(p)
        assert np.isfinite(df.score).all(), p
        for (d, m, k), g in df.groupby(["dataset", "machine", "model"]):
            t = detection_table(g).set_index("id")
            te = g[g.split == "test"]
            gr = [(x[x.label == 0].score.values, x[x.label == 1].score.values) for _, x in te.groupby("id")]
            rng = np.random.default_rng(C.SEED)
            b = np.percentile([mean_auc(gr, rng) for _ in range(1000)], [2.5, 97.5])
            assert abs(mean_auc(gr) - t.auc["mean"]) < 1e-9
            rows.append((d, m, k, len(te), t.auc["mean"], b[0], b[1], t.pauc["mean"], t.drop("mean").auc.min()))
    r = pd.DataFrame(rows, columns="dataset machine model n_test auc ci_low ci_high pauc worst_id_auc".split())
    r.to_csv(C.RESULTS / "detection_summary.csv", index=False)
    print((r.round(4)).to_string())
