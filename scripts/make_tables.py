"""LaTeX tables for rapport/ and Markdown tables for the README, straight from results/*.csv."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C

OUT = C.ROOT / "rapport" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"dense_ae": "AE dense", "conv_ae": "AE conv.", "panns_knn_type": "kNN type",
          "panns_gmm_type": "GMM type", "panns_knn_id": "kNN ID", "panns_gmm_id": "GMM ID"}
FR = {"pump": "Pompe", "fan": "Ventilateur"}


def fr(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def write(name, spec, head, rows):
    """rows: lists of cells, or None for a horizontal rule."""
    esc = lambda c: str(c).replace("_", r"\_").replace("%", r"\%")
    body = [r"\midrule" if r is None else " & ".join(map(esc, r)) + r" \\" for r in rows]
    (OUT / f"{name}.tex").write_text("\n".join(
        [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(map(esc, head)) + r" \\", r"\midrule",
         *body, r"\bottomrule", r"\end{tabular}"]), encoding="utf-8")


def md(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows if r])


def cell(r):
    return f"{fr(100 * r.auc)} / {fr(100 * r.pauc)}"


per = pd.read_csv(C.RESULTS / "detection_per_id.csv").set_index(["dataset", "machine", "id", "model"])
rows = []
for m in C.MACHINES:
    if rows:
        rows.append(None)
    for i in ("id_00", "id_02", "id_04", "id_06", "mean"):
        rows.append([FR[m], "moyenne" if i == "mean" else i] + [cell(per.loc[("dcase", m, i, k)]) for k in MODELS])
head = ["Machine", "ID", *MODELS.values()]
write("dcase", "ll" + "c" * 6, head, rows)
print(md(head, rows), "\n")

rows = []
for m in C.MACHINES:
    if rows:
        rows.append(None)
    for k, lab in MODELS.items():
        rows.append([FR[m], lab] + [cell(per.loc[(f"mimii_{s}dB", m, "mean", k)]) for s in (-6, 0, 6)])
head = ["Machine", "Modèle", "-6 dB", "0 dB", "+6 dB"]
write("noise", "llccc", head, rows)
print(md(head, rows), "\n")

d = pd.read_csv(C.RESULTS / "decision_dcase.csv").set_index(["percentile", "machine", "model"])
rows = []
for m in C.MACHINES:
    if rows:
        rows.append(None)
    for k, lab in MODELS.items():
        for q in (90, 95):
            r = d.loc[(q, m, k)]
            rows.append([FR[m], lab, str(q), *(fr(100 * v) for v in (r.precision, r.recall, r.f1, r.false_alarm_rate, r.balanced_acc))])
    r = d.xs(("always_alarm"), level="model").xs(m, level="machine").iloc[0]
    rows.append([FR[m], "alerte permanente", "-", *(fr(100 * v) for v in (r.precision, r.recall, r.f1, r.false_alarm_rate, r.balanced_acc))])
head = ["Machine", "Modèle", "Perc.", "Précision", "Rappel", "F1", "Fausses al.", "Exact. équil."]
write("decision", "llcccccc", head, rows)
print(md(head, [r for r in rows if r and r[2] != "95"]), "\n")

q = pd.read_csv(C.RESULTS / "edge_quality.csv")
lat = pd.read_csv(C.RESULTS / "edge_latency.csv")
lat = lat[lat.threads == 1].set_index("stage")
V = {"fp32": "FP32", "int8_static": "INT8 statique", "int8_dynamic": "INT8 dynamique"}
rows = []
for (k, v), g in q.groupby(["model", "variant"], sort=False):
    g = g.set_index("machine")
    rows.append(["AE " + k.replace("conv", "conv."), V[v], f"{g.size_kb.mean():.0f}", fr(100 * g.auc["pump"]),
                 fr(100 * g.auc["fan"]), fr(lat.median_ms[f"{k}_{v}"])])
head = ["Modèle", "Version", "Taille (Ko)", "AUC pompe", "AUC vent.", "Latence (ms)"]
write("edge", "llcccc", head, rows)
print(md(head, rows))
print(lat.loc[["wav+log_mel", "panns_cnn14_torch"]])
