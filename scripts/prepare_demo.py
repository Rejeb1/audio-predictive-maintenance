"""Copy what app.py needs to demo/: dense FP32 ONNX models, thresholds, a few example clips.

Examples (by rule): for the machine ID with the best dense AUC, the test normal and the test
fault with the median score - typical clips, not hand-picked successes.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C

OUT = C.ROOT / "demo"
(OUT / "examples").mkdir(parents=True, exist_ok=True)
per_id = pd.read_csv(C.RESULTS / "detection_per_id.csv")
thr, examples = {}, []
for m in C.MACHINES:
    shutil.copy(C.MODELS / "onnx" / f"dcase_{m}_dense_fp32.onnx", OUT / f"{m}.onnx")
    s = pd.read_csv(C.RESULTS / "scores" / f"dcase_{m}_dense.csv")
    thr[m] = {i: {q: float(np.percentile(g.score, q)) for q in (90, 95)}
              for i, g in s[s.split == "val"].groupby("id")}
    q = per_id[(per_id.dataset == "dcase") & (per_id.machine == m) & (per_id.model == "dense_ae")
               & (per_id.id != "mean")]
    best = q.sort_values("auc").id.iloc[-1]
    for label, name in ((0, "normal"), (1, "panne")):
        g = s[(s.split == "test") & (s.id == best) & (s.label == label)].sort_values("score")
        r = g.iloc[len(g) // 2]
        f = f"{m}_{best}_{name}.wav"
        shutil.copy(C.DATA / "dcase" / m / "test" / r.path, OUT / "examples" / f)
        examples.append(dict(file=f, machine=m, id=best, source=r.path, score=r.score))
json.dump({"thresholds": thr, "examples": examples}, open(OUT / "demo.json", "w"), indent=1)
print(json.dumps(examples, indent=1), thr)
