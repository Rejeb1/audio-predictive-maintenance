"""Web demo: anomalous machine sound detection with the dense autoencoder (ONNX, CPU).

/        web page (demo/static/index.html)
/gradio  simple Gradio version of the same analysis
"""
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from scipy.ndimage import uniform_filter
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from asd.config import CLIP_SECONDS, N_MELS, SR
from asd.inference import BAND, AudioError, analyse, band_edges_hz, band_means, log_mel, read_clip

DEMO = ROOT / "demo"
CFG = json.loads((DEMO / "demo.json").read_text())
SESS = {m: ort.InferenceSession(str(DEMO / f"{m}.onnx"), providers=["CPUExecutionProvider"])
        for m in CFG["machines"]}
EXAMPLES = {e["file"]: e for e in CFG["examples"]}
EDGES = band_edges_hz()
MAX_BYTES = 25_000_000
_f = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR / 2)[1:-1]  # band centres
MEL_HZ = _f.round().astype(int).tolist()
MEL_TICKS = [[v, float((np.interp(v, _f, np.arange(N_MELS)) + .5) / N_MELS)] for v in (250, 500, 1000, 2000, 4000, 7000)]

for _s in SESS.values():  # load librosa and the models before the first request
    analyse(_s, log_mel(np.zeros(SR, np.float32)))


def level_of(score, info):
    return "alert" if score > info["p95"] else "watch" if score > info["p90"] else "normal"


def image(v, lo, hi):
    """[mels, T] -> 8-bit pixels, low frequencies in the last row."""
    px = (255 * np.clip((v[::-1] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    return {"w": px.shape[1], "h": px.shape[0], "data": base64.b64encode(px.tobytes()).decode()}


def run(source, machine, mid):
    info = CFG["machines"].get(machine, {}).get(mid)
    if info is None:
        raise AudioError("Unknown machine.")
    y = read_clip(source)
    t0 = time.perf_counter()
    x = log_mel(y)
    score, err = analyse(SESS[machine], x)
    ms = 1e3 * (time.perf_counter() - t0)
    ratio = band_means(err.mean(1)) / np.array(info["ref"])
    top = np.argsort(-ratio)[:3]
    peak = err.mean(0)
    win = np.convolve(peak, np.ones(16) / 16, "valid")  # about half a second
    # error relative to this machine's normal clips, smoothed over 3 mel bands and 0.3 s
    usual = np.interp(np.arange(N_MELS), np.arange(BAND / 2 - .5, N_MELS, BAND), info["ref"])
    rel = uniform_filter(err, (3, 9), mode="nearest") / usual[:, None]
    edges = [[round(a), round(b)] for a, b in EDGES]
    return {
        "score": score, "level": level_of(score, info), "duration": len(y) / SR, "ms": ms,
        "above_normal": float(np.mean(np.array(info["val"]) < score)), "n_val": len(info["val"]),
        "mel_hz": MEL_HZ, "mel_ticks": MEL_TICKS,
        "bands": [{"hz": h, "ratio": float(r)} for h, r in zip(edges, ratio)],
        "top": [{"hz": edges[k], "ratio": float(ratio[k])} for k in top],
        "peak_s": float(win.argmax() * 512 / SR),
        "spec": image(x, float(np.percentile(x, 1)), float(np.percentile(x, 99.5))), "err": image(rel, 1, 4),
    }


app = FastAPI(title="Predictive maintenance from audio", docs_url=None, redoc_url=None)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(DEMO / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config():
    keep = ("p90", "p95", "val", "recall90", "false90")
    return {"machines": {m: {i: {k: d[k] for k in keep} for i, d in v.items()} for m, v in CFG["machines"].items()},
            "examples": [{k: e[k] for k in ("file", "machine", "id", "truth", "clear")} for e in CFG["examples"]],
            "clip_seconds": CLIP_SECONDS}


@app.get("/api/example/{name}")
def example(name: str):
    if name not in EXAMPLES:  # only known names: no path can be injected
        raise HTTPException(404, "Unknown example.")
    return FileResponse(DEMO / "examples" / name, media_type="audio/wav")


@app.post("/api/analyse")
async def api_analyse(request: Request, machine: str, id: str):
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BYTES:
            raise HTTPException(413, "File too large (25 MB maximum).")
    try:
        return await run_in_threadpool(run, io.BytesIO(bytes(body)), machine, id)
    except AudioError as e:
        raise HTTPException(422, str(e)) from e


# ---------- simple Gradio version ----------
def gradio_analyse(path, machine, mid):
    if not path:
        raise gr.Error("Choose or record a sound.")
    try:
        r = run(path, machine.lower(), mid)
    except AudioError as e:
        raise gr.Error(str(e)) from e
    info = CFG["machines"][machine.lower()][mid]
    text = {"normal": "No alert", "watch": "Watch: above the 90th percentile", "alert": "Alert: above the 95th percentile"}
    out = np.frombuffer(base64.b64decode(r["err"]["data"]), np.uint8).reshape(r["err"]["h"], -1)
    return (f"## {text[r['level']]}\n\nScore **{r['score']:.2f}** (90th percentile {info['p90']:.2f}, 95th {info['p95']:.2f}). "
            "An unusual sound is not necessarily a fault, and no alert does not prove that the machine is healthy."), 255 - out


with gr.Blocks(title="Predictive maintenance from audio") as ui:
    gr.Markdown("# Predictive maintenance from audio\nDense autoencoder trained on normal sounds only (DCASE 2020).")
    with gr.Row():
        with gr.Column():
            a = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Sound (10 s)")
            m = gr.Radio(["Pump", "Fan"], value="Pump", label="Machine type")
            i = gr.Dropdown(list(CFG["machines"]["pump"]), value="id_04", label="Machine ID")
            b = gr.Button("Analyse", variant="primary")
        with gr.Column():
            v, im = gr.Markdown(), gr.Image(label="Error map (dark: more error than usual)")
    gr.Examples([[str(DEMO / "examples" / e["file"]), e["machine"].capitalize(), e["id"]] for e in CFG["examples"]], [a, m, i])
    b.click(gradio_analyse, [a, m, i], [v, im])
    for c in (a, m, i):
        c.change(lambda: ("", None), None, [v, im])
app = gr.mount_gradio_app(app, ui, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    # public only on Hugging Face Spaces; on a laptop, reachable from this computer only
    uvicorn.run(app, host="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1", port=int(os.environ.get("PORT", 7860)))
