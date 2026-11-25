"""DCASE metrics and the deployment-style threshold (fitted on normals only)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def auc_pauc(y, s, p=0.1):
    return roc_auc_score(y, s), roc_auc_score(y, s, max_fpr=p)


def detection_table(scores):
    """One row per (model, machine, id), plus the average over ids."""
    rows = []
    test = scores[scores.split == "test"]
    for key, g in test.groupby(["model", "machine", "id"]):
        rows.append((*key, *auc_pauc(g.label, g.score)))
    t = pd.DataFrame(rows, columns=["model", "machine", "id", "auc", "pauc"])
    avg = t.groupby(["model", "machine"], as_index=False)[["auc", "pauc"]].mean()
    return pd.concat([t, avg.assign(id="mean")], ignore_index=True)


def decisions(scores, percentile):
    """Threshold = percentile of held-out normal scores, per model, machine and id."""
    out = []
    for _, g in scores.groupby(["model", "machine", "id"]):
        thr = np.percentile(g[g.split == "val"].score, percentile)
        t = g[g.split == "test"]
        out.append(t.assign(threshold=thr, pred=(t.score > thr).astype(int)))
    return pd.concat(out, ignore_index=True)


def decision_table(scores, percentiles=(90, 95)):
    rows = []
    for q in percentiles:
        for (model, machine), g in decisions(scores, q).groupby(["model", "machine"]):
            p, r, f, _ = precision_recall_fscore_support(
                g.label, g.pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(g.label, g.pred, labels=[0, 1]).ravel()
            rows.append((q, model, machine, p, r, f, fp / (fp + tn), tn, fp, fn, tp))
    return pd.DataFrame(rows, columns=["percentile", "model", "machine", "precision", "recall",
                                       "f1", "false_alarm_rate", "tn", "fp", "fn", "tp"])
