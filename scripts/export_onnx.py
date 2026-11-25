"""Export the DCASE autoencoders to ONNX (FP32, INT8), compare AUC, size and CPU latency.

Calibration uses training normals only. Run with CUDA_VISIBLE_DEVICES=-1.
Outputs: D:/mimii/models/onnx/*, results/edge_quality.csv, results/edge_latency.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C
from asd.autoencoder import DEVICE, AEDetector, patch_starts
from asd.embeddings import CHECKPOINT, load_cnn14
from asd.features import cached_log_mels, load_audio, log_mel
from asd.metrics import detection_table

assert DEVICE == "cpu", "run with CUDA_VISIBLE_DEVICES=-1"
OUT = C.MODELS / "onnx"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 200


def windows(x, det):
    """One clip [mels, frames] -> model input, same windows as AEDetector.score."""
    d = ((x - det.mean.numpy()) / det.std.numpy()).astype(np.float32)
    if det.kind == "dense":
        w = np.lib.stride_tricks.sliding_window_view(d, C.CONTEXT, axis=1)  # [mels, T, 5]
        return w.transpose(1, 2, 0).reshape(w.shape[1], -1).copy()
    return np.stack([d[:, s:s + C.PATCH] for s in patch_starts(d.shape[1])])[:, None]


def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.log_severity_level = 3
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def clip_score(sess, w):
    return float(np.mean((sess.run(None, {"x": w})[0] - w) ** 2))


class Reader(CalibrationDataReader):
    def __init__(self, batches):
        self.it = iter({"x": b} for b in batches)

    def get_next(self):
        return next(self.it, None)


def timing(fn, reps=REPS):
    for _ in range(10):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(1e3 * (time.perf_counter() - t0))
    return np.median(t), np.percentile(t, 95)


def export(det, name):
    fp32, pre = OUT / f"{name}_fp32.onnx", OUT / f"{name}_pre.onnx"
    net = det.net.eval()
    dummy = torch.randn(2, 640) if det.kind == "dense" else torch.randn(2, 1, 128, 64)
    torch.onnx.export(net, (dummy,), str(fp32), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, dynamo=False)
    quant_pre_process(str(fp32), str(pre))  # folds batch-norm into the layers
    return fp32, pre


if __name__ == "__main__":
    quality, latency = [], []
    for m in C.MACHINES:
        df, x = cached_log_mels(f"dcase_{m}", None)
        df["name"] = [Path(p).name for p in df.path]
        train = np.flatnonzero((df.split == "train").values)
        calib = np.random.default_rng(C.SEED).choice(train, 100, replace=False)
        test = np.flatnonzero((df.split == "test").values)
        for kind in ("dense", "conv"):
            name = f"dcase_{m}_{kind}"
            det = AEDetector.load(C.MODELS / f"{name}.pt")
            fp32, pre = export(det, name)
            static, dyn = OUT / f"{name}_int8_static.onnx", OUT / f"{name}_int8_dynamic.onnx"
            quantize_static(str(pre), str(static), Reader([windows(x[i], det) for i in calib]),
                            quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
            files = {"fp32": fp32, "int8_static": static}
            if kind == "dense":
                quantize_dynamic(str(pre), str(dyn), weight_type=QuantType.QInt8)
                files["int8_dynamic"] = dyn
            ref = pd.read_csv(C.RESULTS / "scores" / f"{name}.csv")
            ref = ref[ref.split == "test"].set_index("path").score
            for v, f in files.items():
                s = session(f, 4)
                sc = np.array([clip_score(s, windows(x[i], det)) for i in test])
                t = df.iloc[test].assign(score=sc, model=f"{kind}_{v}")
                r = detection_table(t).query("id == 'mean'").iloc[0]
                dif = np.abs(sc / ref.loc[t.name].values - 1)
                if v == "fp32":
                    assert dif.max() < 1e-3, dif.max()
                quality.append(dict(machine=m, model=kind, variant=v, size_kb=f.stat().st_size / 1024,
                                    auc=r.auc, pauc=r.pauc, median_rel_score_change=np.median(dif)))
                print(quality[-1], flush=True)
                if m == "pump":
                    w = windows(x[test[0]], det)
                    for th in (1, 4):
                        s = session(f, th)
                        a, b = timing(lambda: clip_score(s, w))
                        latency.append(dict(stage=f"{kind}_{v}", threads=th, median_ms=a, p95_ms=b))
            pre.unlink()
    path = df.path[0]
    y = load_audio(path)
    a, b = timing(lambda: log_mel(load_audio(path)))
    latency.append(dict(stage="wav+log_mel", threads=1, median_ms=a, p95_ms=b))
    net = load_cnn14()
    wav = torch.from_numpy(y)[None]
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.no_grad():
            a, b = timing(lambda: net(wav), 30)
        latency.append(dict(stage="panns_cnn14_torch", threads=th, median_ms=a, p95_ms=b))
    q, l = pd.DataFrame(quality), pd.DataFrame(latency)
    q.to_csv(C.RESULTS / "edge_quality.csv", index=False)
    l.to_csv(C.RESULTS / "edge_latency.csv", index=False)
    print(q.round(4).to_string())
    print(l.round(2).to_string())
    print("PANNs checkpoint (MB):", round(CHECKPOINT.stat().st_size / 2**20),
          "| trunk parameters (M):", round(sum(p.numel() for p in net.parameters()) / 1e6, 1))
