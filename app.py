"""Gradio demo: anomalous machine sound detection with the dense autoencoder (ONNX, CPU)."""
import json
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from asd.config import CLIP_SECONDS, CONTEXT, HOP, N_MELS, SR
from asd.features import log_mel

DEMO = ROOT / "demo"
CFG = json.loads((DEMO / "demo.json").read_text())
SESS = {m: ort.InferenceSession(str(DEMO / f"{m}.onnx"), providers=["CPUExecutionProvider"])
        for m in CFG["thresholds"]}
FR = {"Pompe": "pump", "Ventilateur": "fan"}
IDS = ["id_00", "id_02", "id_04", "id_06"]


def read_clip(path):
    """First channel, 16 kHz, at most 10 s. Not padded: silence would look abnormal."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y[:, 0]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y[:SR * CLIP_SECONDS]


def analyse_audio(y, machine):
    """-> (score, log-mel [mels, T], error map [mels, T])"""
    x = log_mel(y)
    w = np.lib.stride_tricks.sliding_window_view(x, CONTEXT, axis=1).transpose(1, 2, 0)  # [T, 5, mels]
    flat = w.reshape(len(w), -1).copy()
    err = (SESS[machine].run(None, {"x": flat})[0] - flat) ** 2
    e = err.reshape(len(w), CONTEXT, N_MELS)
    out, cnt = np.zeros_like(x), np.zeros(x.shape[1])
    for k in range(CONTEXT):
        out[:, k:k + len(w)] += e[:, k].T
        cnt[k:k + len(w)] += 1
    return float(err.mean()), x, out / cnt


def figure(x, err):
    fig = Figure(figsize=(9, 5.2), layout="constrained")
    ax = fig.subplots(2, 1, sharex=True)
    ext = (0, x.shape[1] * HOP / SR, 0, N_MELS)
    a = ax[0].imshow(x, origin="lower", aspect="auto", cmap="viridis", extent=ext)
    b = ax[1].imshow(err, origin="lower", aspect="auto", cmap="Reds", extent=ext, vmin=0,
                     vmax=max(np.percentile(err, 99.5), 100))
    fig.colorbar(a, ax=ax[0], label="dB")
    fig.colorbar(b, ax=ax[1], label="erreur (dB²)")
    ax[0].set(title="Spectrogramme log-mel", ylabel="bande mel")
    ax[1].set(title="Carte d'erreur : zones mal reconstruites", ylabel="bande mel", xlabel="temps (s)")
    return fig


def analyse(path, machine_fr, mid, percentile):
    if not path:
        raise gr.Error("Choisir ou enregistrer un son.")
    try:
        y = read_clip(path)
    except Exception as e:
        raise gr.Error("Fichier audio illisible (WAV ou FLAC conseillé).") from e
    if len(y) < SR:
        raise gr.Error("Son trop court : 1 seconde au minimum.")
    if not np.isfinite(y).all() or np.abs(y).max() < 1e-5:
        raise gr.Error("Son vide ou silencieux.")
    m = FR[machine_fr]
    thr = CFG["thresholds"][m][mid][str(percentile)]
    score, x, err = analyse_audio(y, m)
    alert = score > thr
    txt = (f"## {'⚠️ Alerte : son inhabituel' if alert else '✅ Pas d’alerte'}\n\n"
           f"Score **{score:.2f}**, seuil **{thr:.2f}** ({machine_fr.lower()} {mid}, {percentile}e percentile).")
    if not alert:
        txt += "\n\nL'absence d'alerte ne prouve pas que la machine est saine."
    if len(y) < SR * CLIP_SECONDS:
        txt += f"\n\nSon de {len(y) / SR:.1f} s : le seuil est calibré pour {CLIP_SECONDS} s."
    return txt, figure(x, err)


NOTE = """# Détection non supervisée de pannes par l'audio
Autoencodeur dense entraîné uniquement sur des sons **normaux** (DCASE 2020, pompes et ventilateurs).
Une forte erreur de reconstruction signale un son inhabituel, pas nécessairement une panne.

**Limites.** Au 90e percentile, sur le test DCASE : 41 % des pannes détectées pour 11 % de fausses alertes
(pompe), 28 % pour 14 % (ventilateur). Les seuils ne sont valables que pour ces machines et ces
conditions d'enregistrement : un son d'une autre machine ou d'un autre micro déclenchera souvent une alerte.
"""
SOURCE = """Exemples : clips de test de score médian (normal et panne) issus de DCASE 2020 Task 2
([Zenodo 3678171](https://zenodo.org/records/3678171)), sous-ensemble de MIMII (Purohit et al., 2019),
licence CC BY-NC-SA 4.0. Les noms de fichiers donnent la vérité terrain."""

with gr.Blocks(title="Maintenance prédictive par l'audio") as demo:
    gr.Markdown(NOTE)
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Son (10 s)")
            machine = gr.Radio(list(FR), value="Pompe", label="Type de machine")
            mid = gr.Dropdown(IDS, value="id_04", label="Machine")
            pct = gr.Radio([90, 95], value=90, label="Seuil (percentile des sons normaux)")
            btn = gr.Button("Analyser", variant="primary")
        with gr.Column(scale=2):
            verdict = gr.Markdown()
            plot = gr.Plot(show_label=False)
    inv = {v: k for k, v in FR.items()}
    gr.Examples([[str(DEMO / "examples" / e["file"]), inv[e["machine"]], e["id"], 90] for e in CFG["examples"]],
                [audio, machine, mid, pct])
    gr.Markdown(SOURCE)
    btn.click(analyse, [audio, machine, mid, pct], [verdict, plot])
    for c in (audio, machine, mid, pct):  # never leave a stale verdict on screen
        c.change(lambda: ("", None), None, [verdict, plot])

if __name__ == "__main__":
    demo.launch()
