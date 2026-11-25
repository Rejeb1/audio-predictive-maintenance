"""Copy what app.py needs to demo/: dense FP32 ONNX models, thresholds, references, example clips.

Examples (by rule): for every machine ID, the test normal and the test fault with the
median score (typical), and the fault at the 90th percentile of fault scores (clear).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.features import cached_log_mels
from asd.inference import analyse, band_means

OUT = C.ROOT / "demo"
(OUT / "examples").mkdir(parents=True, exist_ok=True)
per_id = pd.read_csv(C.RESULTS / "detection_per_id.csv").set_index(["dataset", "machine", "model", "id"])
machines, examples = {}, []
for m in C.MACHINES:
    shutil.copy(C.MODELS / "onnx" / f"dcase_{m}_dense_fp32.onnx", OUT / f"{m}.onnx")
    sess = ort.InferenceSession(str(OUT / f"{m}.onnx"), providers=["CPUExecutionProvider"])
    df, x = cached_log_mels(f"dcase_{m}", None)
    s = pd.read_csv(C.RESULTS / "scores" / f"dcase_{m}_dense.csv")
    machines[m] = {}
    for i, g in s.groupby("id"):
        val, test = g[g.split == "val"].score, g[g.split == "test"]
        idx = np.flatnonzero(((df.id == i) & (df.split == "val")).values)
        # mean reconstruction error per frequency region on held-out normal clips
        ref = np.mean([band_means(analyse(sess, x[k])[1].mean(1)) for k in idx], axis=0)
        d = {"p90": float(np.percentile(val, 90)), "p95": float(np.percentile(val, 95)),
             "val": sorted(round(float(v), 4) for v in val), "ref": [round(float(v), 4) for v in ref],
             "train_clips": int(((df.id == i) & (df.split == "train")).sum()),
             "auc": float(per_id.auc[("dcase", m, "dense_ae", i)])}
        for q in (90, 95):
            p = test.score > d[f"p{q}"]
            d[f"recall{q}"] = float(p[test.label == 1].mean())
            d[f"false{q}"] = float(p[test.label == 0].mean())
        machines[m][i] = d
        for label, name, q in ((0, "normal", .5), (1, "fault", .5), (1, "clear_fault", .9)):
            t = test[test.label == label].sort_values("score")
            r = t.iloc[int(len(t) * q)]
            f = f"{m}_{i}_{name}.wav"
            shutil.copy(C.DATA / "dcase" / m / "test" / r.path, OUT / "examples" / f)
            examples.append(dict(file=f, machine=m, id=i, truth=name.split("_")[-1], clear=q == .9, source=r.path, score=float(r.score)))
json.dump({"machines": machines, "examples": examples}, open(OUT / "demo.json", "w"), indent=1)
print(json.dumps(examples, indent=1))
