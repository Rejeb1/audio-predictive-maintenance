"""Tables and figures (French) from results/scores -> results/*.csv, results/figures/*.png

Run scripts/noise_table.py first (bootstrap summary).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.metrics import decision_table, decisions, detection_table

FIG = C.RESULTS / "figures"
STYLE = {  # model: (label, colour, marker)
    "dense_ae": ("AE dense", "#2a78d6", "o"),
    "conv_ae": ("AE convolutif", "#eb6834", "s"),
    "panns_knn_type": ("PANNs kNN (type)", "#1baf7a", "^"),
    "panns_gmm_type": ("PANNs GMM (type)", "#eda100", "v"),
    "panns_knn_id": ("PANNs kNN (ID)", "#e87ba4", "D"),
    "panns_gmm_id": ("PANNs GMM (ID)", "#008300", "P"),
}
# DCASE 2020 Task 2 development set, published baseline (mean over 10 runs): AUC, pAUC
OFFICIAL = {"pump": (0.7289, 0.5999), "fan": (0.6583, 0.5245)}
FR = {"pump": "Pompe", "fan": "Ventilateur"}
INK, MUTED = "#0b0b0b", "#52514e"
BOX = dict(facecolor="#fcfcfb", edgecolor="none", pad=1)
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                     "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def load_scores():
    files = [p for p in sorted((C.RESULTS / "scores").glob("*.csv")) if not p.stem.endswith("_history")]
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def fig_dcase(summary):
    d = summary[summary.dataset == "dcase"].set_index(["machine", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, layout="constrained")
    for ax, m in zip(axes, C.MACHINES):
        for i, k in enumerate(STYLE):
            r = d.loc[(m, k)]
            ax.plot([r.ci_low, r.ci_high], [i, i], color="#2a78d6", lw=2)
            ax.plot(r.auc, i, "o", ms=9, color="#2a78d6", mec="#fcfcfb")
            ax.text(r.ci_high + .008, i, f"{100 * r.auc:.1f}", va="center", bbox=BOX)
        for v, lab in ((OFFICIAL[m][0], f"baseline officielle {100 * OFFICIAL[m][0]:.1f}"), (.5, "hasard")):
            ax.axvline(v, color=MUTED, ls="--", lw=1, zorder=0)
            ax.text(v, -.7, lab, ha="center", color=MUTED, fontsize=8, bbox=BOX)
        ax.set(yticks=range(6), yticklabels=[s[0] for s in STYLE.values()], xlim=(.42, .88),
               xlabel="AUC moyenne sur 4 machines", ylim=(5.6, -1))
        ax.set_title(FR[m], loc="left", weight="bold")
    fig.suptitle("DCASE 2020 : AUC et intervalle bootstrap à 95 %", fontsize=10)
    fig.savefig(FIG / "dcase_auc.png", dpi=180)


def fig_noise(summary):
    snr = {"mimii_-6dB": -6, "mimii_0dB": 0, "mimii_6dB": 6}
    for metric, lab in (("auc", "AUC"), ("pauc", "pAUC (FPR ≤ 0,1)")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, layout="constrained")
        for ax, m in zip(axes, C.MACHINES):
            for i, (k, (name, col, mk)) in enumerate(STYLE.items()):
                g = summary[(summary.machine == m) & (summary.model == k) & summary.dataset.isin(snr)]
                g = g.assign(x=g.dataset.map(snr)).sort_values("x")
                x = g.x + (i - 2.5) * .18  # dodge so that intervals stay readable
                if metric == "auc":
                    ax.vlines(x, g.ci_low, g.ci_high, color=col, lw=1, alpha=.6)
                ax.plot(x, g[metric], color=col, marker=mk, lw=2, ms=7, mec="#fcfcfb", label=name)
            ax.axhline(.5, color=MUTED, ls="--", lw=1)
            ax.set(xticks=[-6, 0, 6], xticklabels=["−6", "0", "+6"], xlabel="RSB (dB)", ylim=(.4, 1))
            ax.set_title(FR[m], loc="left", weight="bold")
            ax.grid(axis="y", color="#e6e5e0", lw=.8)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(lab)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=6,
                   frameon=False, fontsize=8.5)
        fig.suptitle(f"MIMII (sous-ensemble) : {lab} selon le bruit, un modèle entraîné par RSB", fontsize=10)
        fig.savefig(FIG / f"noise_{metric}.png", dpi=180)


def fig_confusion(dcase):
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6), layout="constrained")
    dec = decisions(dcase, 90)
    for row, m in zip(axes, C.MACHINES):
        for ax, k in zip(row, ("dense_ae", "conv_ae", "panns_gmm_id")):
            g = dec[(dec.machine == m) & (dec.model == k)]
            cm = confusion_matrix(g.label, g.pred, labels=[0, 1])
            pct = cm / cm.sum(1, keepdims=True)
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}\n{100 * pct[i, j]:.0f} %", ha="center", va="center",
                            color="white" if pct[i, j] > .55 else INK)
            ax.set(xticks=[0, 1], xticklabels=["normal", "alerte"], yticks=[0, 1],
                   yticklabels=["normal", "panne"])
            ax.tick_params(length=0)
            ax.set_title(f"{FR[m]} : {STYLE[k][0]}", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("réalité")
    fig.suptitle("DCASE : seuil au 90e percentile des normaux de validation", fontsize=10)
    fig.savefig(FIG / "confusion_dcase.png", dpi=180)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    s = load_scores()
    n = s.groupby(["dataset", "machine", "model"]).ngroups
    assert n == 48, n
    assert not s.duplicated(["dataset", "machine", "model", "id", "split", "path", "label"]).any()
    summary = pd.read_csv(C.RESULTS / "detection_summary.csv")
    assert len(summary) == 48, "run scripts/noise_table.py first"
    per_id = pd.concat([detection_table(g).assign(dataset=d) for d, g in s.groupby("dataset")])
    per_id.to_csv(C.RESULTS / "detection_per_id.csv", index=False)
    dcase = s[s.dataset == "dcase"]
    t = decision_table(dcase)
    always = []  # an alarm on every clip: reference for F1, which the share of faults inflates
    for m, g in dcase[(dcase.split == "test") & (dcase.model == "dense_ae")].groupby("machine"):
        p = g.label.mean()
        always.append(dict(percentile=np.nan, model="always_alarm", machine=m, precision=p, recall=1,
                           f1=2 * p / (1 + p), false_alarm_rate=1, tn=0, fp=(g.label == 0).sum(),
                           fn=0, tp=g.label.sum()))
    t = pd.concat([t, pd.DataFrame(always)])
    t["balanced_acc"] = (t.recall + 1 - t.false_alarm_rate) / 2
    t.to_csv(C.RESULTS / "decision_dcase.csv", index=False)
    print(t.round(3).to_string())
    print(per_id[(per_id.dataset == "dcase") & (per_id.model == "dense_ae")].round(4).to_string())
    fig_dcase(summary)
    fig_noise(summary)
    fig_confusion(dcase)
