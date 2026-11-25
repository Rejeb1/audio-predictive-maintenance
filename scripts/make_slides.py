"""Build presentation/index.html (English, 5 speakers) on the "Capsule" template (MIT, Zara Zhang).

The template's stylesheet and navigation script are reused as shipped; numbers come from results/*.csv.
"""
import base64
import io
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asd import config as C

P = C.ROOT / "presentation"
A = P / "assets"
TPL = (A / "capsule_template.html").read_text(encoding="utf-8")
S = pd.read_csv(C.RESULTS / "detection_summary.csv").set_index(["dataset", "machine", "model"])
ID = pd.read_csv(C.RESULTS / "detection_per_id.csv").set_index(["dataset", "machine", "id", "model"])
DEC = pd.read_csv(C.RESULTS / "decision_dcase.csv").set_index(["percentile", "machine", "model"])
Q = pd.read_csv(C.RESULTS / "edge_quality.csv").set_index(["machine", "model", "variant"])
L = pd.read_csv(C.RESULTS / "edge_latency.csv")
L = L[L.threads == 1].set_index("stage").median_ms

TEAM = [("Mohamed Aziz Ben Rejeb", "coral"), ("Wael Znagui", "lime"), ("Maram Nuigues", "sky"),
        ("Khalil Znazen", "violet"), ("Aymen Turki", "yellow")]
PARTS = ["Context and data", "Autoencoders", "Pretrained embeddings", "Results and robustness",
         "Explainability and deployment"]
MODELS = [("dense_ae", "Dense AE"), ("conv_ae", "Conv AE"), ("panns_knn_type", "PANNs kNN, type"),
          ("panns_gmm_type", "PANNs GMM, type"), ("panns_knn_id", "PANNs kNN, ID"), ("panns_gmm_id", "PANNs GMM, ID")]
OFFICIAL = {"pump": 72.89, "fan": 65.83}


def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def img(path, crop=None, width=1500):
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    f = io.BytesIO()
    im.save(f, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(f.getvalue()).decode()


slides = []


def slide(cls, body, part=None):
    n = len(slides) + 1
    tag = ""
    if part is not None:
        name, col = TEAM[part]
        tag = (f'<div class="speaker"><span class="pill pill-{col}">{part + 1}. {PARTS[part]}</span>'
               f'<span class="who">{name}</span></div>')
    slides.append(f'<div class="slide {cls}{" active" if n == 1 else ""}" data-slide="{n}">{body}{tag}</div>')


def head(title, sub=""):
    s = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="slide-header x-head"><h2>{title}</h2>{s}</div>'


def deco(cls, item, pills):
    return f'<div class="{cls}">' + "".join(
        f'<div class="{item} pill-{c}" style="padding:.55em 1.3em;{pos};transform:rotate({r}deg);font-size:.85rem">{t}</div>'
        for t, c, pos, r in pills) + "</div>"


def cards(items, cols=3):
    return f'<div class="cards-grid" style="grid-template-columns:repeat({cols},1fr)">' + "".join(
        f'<div class="pillar-card"><div class="card-icon pill-{c}">{i}</div><h3>{t}</h3><p>{p}</p></div>'
        for i, c, t, p in items) + "</div>"


def stats(items):
    return f'<div class="stats-grid" style="grid-template-columns:repeat({len(items)},1fr)">' + "".join(
        f'<div class="stat-pill"><div class="stat-number" style="color:var(--{c}-ink)">{n}</div>'
        f'<div class="stat-label">{t}</div><div class="stat-bar pill-{c}"></div></div>' for n, t, c in items) + "</div>"


def BG(c):
    return "card" if c == "white" else c


def flow(nodes):
    return '<div class="diagram-container">' + '<div class="diagram-connector"></div>'.join(
        f'<div class="diagram-node" style="background:var(--{BG(c)})">{t}</div>' for t, c in nodes) + "</div>"


def notes(items):
    return '<div class="notes-row">' + "".join(
        f'<div><div class="pill pill-{c}">{t}</div><p>{p}</p></div>' for t, c, p in items) + "</div>"


# ---------- charts (inline SVG, values from CSV) ----------
def forest(m):
    w, h, x0, x1 = 560, 330, 170, 545
    X = lambda v: x0 + (v - 45) / 40 * (x1 - x0)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="AUC by method, {m}">']
    for v in (50, 60, 70, 80):
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="30" y2="295" stroke="#dddbd2"/>'
                 f'<text x="{X(v):.1f}" y="316" text-anchor="middle" class="ax">{v}</text>')
    v = OFFICIAL[m]
    o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="30" y2="295" stroke="#1A1A1A" stroke-width="1.6" stroke-dasharray="6 4"/>'
             f'<text x="{X(v):.1f}" y="20" text-anchor="middle" class="ax">official baseline {v:.1f}</text>')
    best = max(MODELS, key=lambda k: S.auc[("dcase", m, k[0])])[0]
    for i, (k, lab) in enumerate(MODELS):
        r, y = S.loc[("dcase", m, k)], 55 + i * 44
        c = "var(--coral)" if k == best else "var(--sky-ink)"
        b = ' font-weight="700"' if k == best else ""
        o.append(f'<g><title>{lab}: AUC {100 * r.auc:.1f} (95% interval {100 * r.ci_low:.1f} to {100 * r.ci_high:.1f})</title>'
                 f'<rect x="0" y="{y - 20}" width="{w}" height="40" fill="transparent"/>'
                 f'<text x="{x0 - 14}" y="{y + 5}" text-anchor="end" class="lb"{b}>{lab}</text>'
                 f'<line x1="{X(100 * r.ci_low):.1f}" x2="{X(100 * r.ci_high):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="3"/>'
                 f'<circle cx="{X(100 * r.auc):.1f}" cy="{y}" r="7" fill="{c}" stroke="#FCFBF6" stroke-width="2"/>'
                 f'<text x="{X(100 * r.ci_high) + 8:.1f}" y="{y + 5}" class="lb halo"{b}>{100 * r.auc:.1f}</text></g>')
    return "".join(o) + "</svg>"


LINES = [("dense_ae", "Dense AE", "#2D5FBF", "circle", ""), ("conv_ae", "Conv AE", "#E85D4E", "square", ""),
         ("panns_knn_id", "PANNs kNN, ID", "#7A3FD1", "diamond", ""), ("panns_gmm_id", "PANNs GMM, ID", "#1A1A1A", "triangle", ""),
         ("panns_knn_type", "PANNs kNN, type", "#8d8c86", "circle", "7 4"), ("panns_gmm_type", "PANNs GMM, type", "#8d8c86", "square", "2 4")]


def marker(kind, x, y, c):
    p = {"square": f"M{x - 6} {y - 6}h12v12h-12z", "diamond": f"M{x} {y - 8}l8 8l-8 8l-8 -8z",
         "triangle": f"M{x} {y - 8}l8 14h-16z"}.get(kind)
    st = f'fill="{c}" stroke="#FCFBF6" stroke-width="2"'
    return f'<path d="{p}" {st}/>' if p else f'<circle cx="{x}" cy="{y}" r="7" {st}/>'


def noise(m):
    w, h = 520, 330
    X = {-6: 80, 0: 280, 6: 480}
    Y = lambda v: 290 - (v - 45) / 55 * 265
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="AUC against noise, {m}">']
    for v in (50, 60, 70, 80, 90, 100):
        o.append(f'<line x1="50" x2="510" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dddbd2"/>'
                 f'<text x="42" y="{Y(v) + 5:.1f}" text-anchor="end" class="ax">{v}</text>')
    for s, x in X.items():
        lab = {-6: "−6 dB", 0: "0 dB", 6: "+6 dB"}[s]
        o.append(f'<text x="{x}" y="318" text-anchor="middle" class="lb">{lab}</text>')
    for k, lab, c, mk, dash in reversed(LINES):
        pts = [(X[s], round(Y(100 * S.auc[(f"mimii_{s}dB", m, k)]), 1), s) for s in X]
        o.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y, _ in pts)}" fill="none" stroke="{c}" '
                 f'stroke-width="3" stroke-dasharray="{dash}"/>')
        for x, y, s in pts:
            r = S.loc[(f"mimii_{s}dB", m, k)]
            o.append(f'<g><title>{lab}, {s:+d} dB: {100 * r.auc:.1f} (95% interval {100 * r.ci_low:.1f} to '
                     f'{100 * r.ci_high:.1f})</title><circle cx="{x}" cy="{y}" r="14" fill="transparent"/>{marker(mk, x, y, c)}</g>')
    return "".join(o) + "</svg>"


def noise_legend():
    return '<div class="legend">' + "".join(
        f'<span><svg viewBox="0 0 44 20"><line x1="2" x2="42" y1="10" y2="10" stroke="{c}" stroke-width="3" '
        f'stroke-dasharray="{d}"/>{marker(mk, 22, 10, c)}</svg>{lab}</span>' for _, lab, c, mk, d in LINES) + "</div>"


def heat():
    def col(v):
        t = max(0, min(1, (v - 50) / 45))
        a, b = (252, 251, 246), (139, 180, 247)
        return "rgb(" + ",".join(str(round(p + (q - p) * t)) for p, q in zip(a, b)) + ")"
    o = ['<table class="tbl heat"><tr><th></th><th></th>' + "".join(f"<th>{lab}</th>" for _, lab in MODELS) + "</tr>"]
    for m in ("pump", "fan"):
        for j, i in enumerate(("id_00", "id_02", "id_04", "id_06")):
            v = [100 * ID.auc[("dcase", m, i, k)] for k, _ in MODELS]
            first = f'<th rowspan="4" class="rowh">{m.capitalize()}</th>' if j == 0 else ""
            o.append(f"<tr>{first}<th>{i.replace('id_', 'ID ')}</th>" + "".join(
                f'<td style="background:{col(x)};{"font-weight:700" if x == max(v) else ""}">{x:.1f}</td>' for x in v) + "</tr>")
    return "".join(o) + "</table>"


# ---------- slides ----------
names = "".join(f'<span class="pill pill-{c}">{n}</span>' for n, c in TEAM)
slide("slide-1", deco("deco-pills", "deco-pill", [
    ("Pumps", "coral", "top:9%;left:9%", -12), ("Log-mel", "lavender", "top:12%;right:11%", 8),
    ("No fault labels", "sky", "top:30%;left:4%", 6), ("AUC", "peach", "top:34%;right:7%", -8),
    ("Fans", "lime", "bottom:8%;left:46%", 4)]) +
    '<div class="title-pill">Deep Learning mini-project &middot; IIA5 &middot; INSAT</div>'
    '<h1 class="main-title">Hearing faults<br>in machine noise<span class="subtitle">Predictive maintenance from audio: '
    'unsupervised detection of machine faults</span></h1>'
    f'<div class="team">{names}</div><p class="sup">Supervisor: Ridha ALOUI &middot; December 2026</p>')

toc = "".join(
    f'<div class="toc-row"><div class="card-icon pill-{TEAM[i][1]}">{i + 1}</div><div class="toc-t"><h3>{p}</h3><p>{d}</p></div>'
    f'<span class="pill pill-{TEAM[i][1]}">{TEAM[i][0]}</span></div>'
    for i, (p, d) in enumerate(zip(PARTS, (
        "The problem, the pipeline and the two datasets", "Learning to rebuild normal sound",
        "PANNs, test protocol and overall results", "Per machine, under noise, with an alert threshold",
        "Error maps, ONNX and INT8, demo, conclusion"))))
slide("slide-3", f'<div class="slide-inner x-tight">{head("Table of contents")}<div class="toc">{toc}</div></div>')

# 1 Context and data
slide("slide-2", '<div class="slide-inner"><div class="left-content"><div class="accent-line"></div>'
      "<h2>A failing machine often sounds different first</h2>"
      "<p>Worn bearings, leaks and imbalance change a machine's acoustic signature. Today, spotting this relies on "
      "experienced technicians or costly vibration sensors.</p>"
      "<p><b>Question:</b> can we detect anomalies from plain audio, learning <b>only from normal sounds</b>, "
      "despite factory noise?</p></div>"
      '<div class="right-visual"><div class="orbit-center">Normal</div>' + "".join(
          f'<div class="orbit-pill pill-{c}" style="padding:.6em 1.3em;font-size:.85rem;{pos};transform:rotate({r}deg)">{t}</div>' for t, c, pos, r in (
              ("Leak", "coral", "top:8%;left:16%", -14), ("Imbalance", "lavender", "top:4%;right:8%", 10),
              ("Worn bearing", "yellow", "bottom:10%;left:4%", 8), ("Unknown fault", "sky", "bottom:6%;right:4%", -8))) +
      "</div></div>", 0)
slide("slide-8", f'<div class="slide-inner x-wide">{head("From a sound to an alert")}' + flow([
    ("Audio<br>10 s, 16 kHz", "white"), ("Log-mel<br>spectrogram", "lavender"), ("Model of<br>normal sound", "lime"),
    ("Anomaly<br>score", "sky"), ("Threshold", "yellow"), ("Verdict and<br>error map", "coral")]) + notes([
        ("Learn", "lime", "Train on normal sounds only: no fault is ever shown to the model"),
        ("Compare", "sky", "Three approaches, two machine types, three noise levels"),
        ("Explain", "coral", "Show which frequencies and moments look abnormal")]) + "</div>", 0)
slide("slide-7", f'<div class="slide-inner">{head("Two datasets, two questions")}' +
      '<div class="two"><div class="panel"><h3>DCASE 2020 Task 2: main comparison</h3>' +
      '<table class="tbl"><tr><th></th><th>Train (normal)</th><th>Test normal</th><th>Test fault</th></tr>'
      "<tr><th>Pump</th><td>3,349</td><td>400</td><td>456</td></tr><tr><th>Fan</th><td>3,675</td><td>400</td><td>1,475</td></tr></table>"
      "<p>4 machines (IDs) per type. 10% of the training normals are held out to set thresholds.</p></div>"
      '<div class="panel"><h3>MIMII subset: noise study</h3><table class="tbl"><tr><th>Per machine and noise level</th><th>Clips</th></tr>'
      "<tr><th>Normal: train / held out / test</th><td>135 / 15 / 50</td></tr><tr><th>Faults (test only)</th><td>50</td></tr></table>"
      "<p>Same clips at −6, 0 and +6 dB. New models per level: DCASE comes from MIMII, so reusing them would leak data.</p></div></div></div>", 0)

# 2 Autoencoders
slide("slide-3", f'<div class="slide-inner">{head("Three ways to model normal sound")}' + cards([
    ("I", "coral", "Dense autoencoder", "Rebuilds 5 short frames at a time. Same design as the official DCASE baseline."),
    ("II", "lime", "Convolutional autoencoder", "Treats the spectrogram as an image and rebuilds time-frequency patches."),
    ("III", "sky", "Pretrained embeddings", "PANNs describes each clip; we measure the distance to normal clips.")]) + "</div>", 1)
slide("slide-8", f'<div class="slide-inner x-wide">{head("Rebuild badly = suspicious", "Trained on normal sounds, an autoencoder should fail to rebuild what it has never heard")}' + flow([
    ("Input<br>640", "white"), ("Encoder<br>4 x 128", "lavender"), ("Bottleneck<br>8", "coral"), ("Decoder<br>4 x 128", "lavender"),
    ("Output<br>640", "white")]) + notes([
        ("Input", "white", "5 frames x 128 mel bands"), ("Training", "lime", "100 epochs, Adam 1e-3, one model per machine type"),
        ("Score", "yellow", "Mean squared reconstruction error over the clip")]) + "</div>", 1)
slide("slide-9", '<div class="slide-inner"><div class="visual-side"><div class="visual-frame"><div class="frame-pattern"></div>'
      '<div class="stack">' + "".join(f'<div class="pill pill-{c}" style="width:{w}%">{t}</div>' for t, c, w in (
          ("Patch 128 x 64", "white", 88), ("4 strided convolutions", "lavender", 70), ("Bottleneck 40", "coral", 44),
          ("4 transposed convolutions", "lavender", 70), ("Rebuilt patch", "white", 88))) +
      "</div></div></div>"
      '<div class="text-side"><h2>Convolutional autoencoder</h2><p>Each mel band is standardised, then the network sees '
      "128 x 64 patches: about 2 seconds of context instead of 5 frames.</p><p>80 epochs, OneCycle learning rate up to 2e-3.</p>"
      '<div class="text-pills"><div class="mini-pill pill-lime">More context</div><div class="mini-pill pill-sky">About 1.5 MB</div>'
      '<div class="mini-pill pill-yellow">Trained on Kaggle T4</div></div></div></div>', 1)

# 3 Embeddings
slide("slide-8", f'<div class="slide-inner x-wide">{head("PANNs: a network that already knows sound")}' + flow([
    ("10 s clip", "white"), ("CNN14, pretrained<br>on AudioSet", "lavender"), ("2,048 numbers", "lime"), ("kNN or GMM", "sky"), ("Score", "coral")]) + notes([
        ("kNN", "sky", "Mean distance to the 5 nearest normal clips"),
        ("GMM", "violet", "How unlikely the clip is under a model of normal clips"),
        ("Per type", "lime", "Reference: all 4 machines, like the autoencoders"),
        ("Per ID", "yellow", "Reference: that exact machine, so its ID must be known")]) + "</div>", 2)
slide("slide-6", f'<div class="slide-inner">{head("How we kept the comparison honest")}<div class="timeline"><div class="timeline-track"></div>' + "".join(
    f'<div class="timeline-step"><div class="step-node pill-{c}">{i}</div><div class="step-label">{t}</div><div class="step-desc">{d}</div></div>'
    for i, (t, c, d) in enumerate((
        ("No tuning on test", "coral", "Settings fixed in advance, all variants reported"),
        ("Separate data", "lime", "Thresholds use held-out normal clips only"),
        ("AUC and pAUC", "sky", "Official metrics, averaged over 4 machines"),
        ("95% intervals", "violet", "Bootstrap over test clips; one training run per model"),
        ("Training checks", "yellow", "A faulty laptop GPU was caught; runs redone on Kaggle")), 1)) + "</div></div>", 2)
slide("slide-4 chartslide", f'<div class="slide-inner x-wide">{head("No single winner on DCASE", "Mean AUC (%) and 95% interval; 50 is chance. Best in coral")}'
      f'<div class="two"><div class="panel"><h3>Pump</h3>{forest("pump")}</div><div class="panel"><h3>Fan</h3>{forest("fan")}</div></div>'
      '<p class="take">Our dense AE matches the official baseline. PANNs GMM per ID leads on pump, but its interval overlaps the dense AE.</p></div>', 2)

# 4 Results
slide("slide-4 chartslide", f'<div class="slide-inner x-wide">{head("The machine matters as much as the method", "DCASE AUC (%) per machine. Bold: best in the row")}'
      f'<div class="panel">{heat()}</div><p class="take">Conv AE: 93.3 on pump ID 04, but 48.8 (chance) on ID 02. Fan ID 00 is hard for every method.</p></div>', 3)
slide("slide-4 chartslide", f'<div class="slide-inner x-wide">{head("Noise is a major limit", "MIMII subset: mean AUC (%) against signal-to-noise ratio")}'
      f'<div class="two"><div class="panel"><h3>Pump</h3>{noise("pump")}</div><div class="panel"><h3>Fan</h3>{noise("fan")}</div></div>{noise_legend()}'
      '<p class="take">From −6 to +6 dB, every method gains 8 to 30 points. The dense AE reaches 94.3 on fan at +6 dB.</p></div>', 3)


def drow(m, k, lab):
    r = DEC.loc[(90, m, k)]
    return (f"<tr><th>{m.capitalize()}</th><th>{lab}</th>" + "".join(
        f"<td>{100 * v:.0f}%</td>" for v in (r.precision, r.recall)) + f"<td>{r.f1:.2f}</td>" + "".join(
        f"<td>{100 * v:.0f}%</td>" for v in (r.false_alarm_rate, r.balanced_acc)) + "</tr>")


slide("slide-4 chartslide", f'<div class="slide-inner x-wide">{head("With a real alert threshold, most faults are missed", "DCASE, threshold at the 90th percentile of held-out normal scores")}'
      '<div class="panel"><table class="tbl big"><tr><th></th><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>False alarms</th><th>Balanced accuracy</th></tr>'
      + "".join(drow(m, k, lab) for m in ("pump", "fan") for k, lab in (MODELS[0], MODELS[1], MODELS[5])) +
      "</table></div>" + notes([
          ("Missed faults", "coral", "Dense AE misses 59% on pump and 72% on fan"),
          ("F1 can mislead", "yellow", "Alarming on every clip scores 0.69 on pump and 0.88 on fan, yet is useless"),
          ("Target missed", "lavender", "False alarms often exceed the 10% target")]) + "</div>", 3)

# 5 Explainability and deployment
slide("slide-9 mapslide", f'<div class="slide-inner x-map"><div class="panel"><img alt="Pump error maps" src="{img(C.RESULTS / "figures" / "error_map_pump_en.png")}"></div>'
      '<div class="text-side"><h2>Error maps show where</h2><p><b>Middle:</b> the error follows the moving tones of this pump fault.</p>'
      "<p><b>Right:</b> this fault sounds normal to the model, so it is missed.</p>"
      "<p><b>Caution:</b> a high score means unusual, not necessarily broken. On one fan clip, short tones that may be factory noise raise the score.</p></div></div>", 4)
e = lambda m, k, v: f"{100 * Q.auc[(m, k, v)]:.1f}"
kb = lambda k, v: f"{Q.size_kb[('pump', k, v)]:,.0f} KB"
rows = "".join(f"<tr><th>{lab}</th><td>{kb(k, v)}</td><td>{e('pump', k, v)}</td><td>{e('fan', k, v)}</td><td>{L[k + '_' + v]:.1f} ms</td></tr>"
               for k, v, lab in (("dense", "fp32", "Dense FP32"), ("dense", "int8_dynamic", "Dense INT8 dynamic"),
                                 ("dense", "int8_static", "Dense INT8 static"), ("conv", "fp32", "Conv FP32"), ("conv", "int8_static", "Conv INT8 static")))
slide("slide-4 chartslide", f'<div class="slide-inner x-wide">{head("Small enough for the edge", "ONNX Runtime, laptop i5-10300H, 1 thread, median per 10 s clip; not tested on an embedded board")}'
      f'<div class="panel"><table class="tbl big"><tr><th></th><th>Size</th><th>Pump AUC</th><th>Fan AUC</th><th>Model time</th></tr>{rows}</table></div>' +
      notes([("3.5x smaller", "lime", "Dense INT8 dynamic keeps its AUC"),
             ("About 8 ms", "sky", "Whole dense INT8 pipeline, log-mel included"),
             ("820 ms", "coral", "PANNs CNN14 in PyTorch: about 100 times slower, 80M parameters")]) + "</div>", 4)
slide("slide-9", f'<div class="slide-inner x-demo"><div class="panel"><img alt="Demo screenshot" src="{img(A / "demo.jpg")}"></div>'
      '<div class="text-side"><h2>Live demo</h2><p>Upload or record a sound, choose the machine, and get a plain verdict, the score among healthy clips, and the error over the spectrogram.</p>'
      "<p>Each machine has a healthy clip, a typical fault and a clear fault. The typical fault is <b>missed on 7 of 8 machines</b>.</p>"
      '<div class="text-pills"><div class="mini-pill pill-lime">FastAPI + Gradio</div><div class="mini-pill pill-sky">ONNX on CPU</div>'
      '<div class="mini-pill pill-yellow">localhost:7860</div></div></div></div>', 4)
slide("slide-3", f'<div class="slide-inner">{head("What we learned, and what is missing")}' + cards([
    ("1", "lime", "Findings", "Normal sounds alone can reveal faults. The best method depends on the machine, and noise is a major limit. "
     "The larger models did not always win."),
    ("2", "coral", "Limits", "Most faults are missed at our threshold. Two machine types, one training run per model, "
     "and no other microphone or factory tested."),
    ("3", "sky", "Next steps", "One autoencoder per machine, scoring the most abnormal frames, self-supervision with machine IDs, "
     "and recalibration on site.")]) + '<p class="src">Data: DCASE 2020 Task 2 (Koizumi et al., Zenodo 3678171, CC BY-NC-SA 4.0); '
     "MIMII (Purohit et al., 2019, Zenodo 3384388, CC BY-SA 4.0). Model: PANNs (Kong et al., 2020).</p></div>", 4)
slide("slide-10", deco("deco-pills-closing", "c-pill", [
    ("Pumps", "lime", "top:14%;left:10%", -12), ("Fans", "sky", "top:18%;right:12%", 12), ("Noise", "lavender", "bottom:14%;left:12%", 10),
    ("Demo", "coral", "bottom:16%;right:13%", -8)]) +
    '<div class="closing-content"><div class="closing-pill">Questions?</div><h2>Thank you for listening</h2><div class="closing-line"></div>'
    f'<div class="team">{names}</div></div>')

# ---------- assemble ----------
style = re.search(r"<style>(.*?)</style>", TPL, re.S)[1]
style = re.sub(r"@media \(max-width: \d+px\)", "@media (max-width: 1px)", style)  # rem scales with the viewport instead
script = re.search(r"<script>(.*?)</script>", TPL, re.S)[1]
fonts = "".join(f"@font-face{{font-family:'{f}';font-style:{s};font-weight:{w};src:url({b64(A / n, 'font/woff2')}) format('woff2')}}"
                for f, s, w, n in (("Bodoni Moda", "italic", "600 800", "font1.woff2"), ("Bodoni Moda", "normal", "600 800", "font2.woff2"),
                                   ("Space Grotesk", "normal", "400 700", "font3.woff2")))
EXTRA = """
  /* Project additions: same fonts, palette and components, sized for projection */
  html { font-size: min(1.4vw, 2.5vh); }
  .slide { padding: 2.2rem 4rem 3.6rem; }
  .slide .slide-inner { max-width: 60rem; }
  .slide .slide-inner.x-wide { max-width: 64rem; gap: 1.6rem; }
  .slide .slide-inner.x-tight { gap: 1rem; }
  .slide .x-head { text-align: center; margin: 0; }
  .slide .x-head h2 { font-size: 2.5rem; line-height: 1.1; font-family: var(--font-display); font-weight: 700; }
  .x-head .sub { font-size: 1.05rem; margin-top: .5rem; opacity: .85; }
  .nav-hint { display: none; } .slide-counter { font-size: .85rem; }
  .nav-dots { gap: .45rem; right: 1.2rem; } .nav-dot { width: 9px; height: 9px; }
  .speaker { position: absolute; left: 2rem; bottom: 1.3rem; display: flex; gap: .8rem; align-items: center; }
  .speaker .pill { padding: .25rem .9rem; font-size: .8rem; font-weight: 600; } .speaker .who { font-size: .95rem; font-weight: 600; }
  .team { display: flex; flex-wrap: wrap; justify-content: center; gap: .7rem; margin-top: 2rem; position: relative; z-index: 2; max-width: 56rem; }
  .team .pill { padding: .5rem 1.2rem; font-size: 1rem; font-weight: 600; box-shadow: 4px 4px 0 var(--shadow); }
  .sup { margin-top: 1.4rem; font-size: 1rem; position: relative; z-index: 2; }
  .slide-1 .main-title { font-size: 5rem; line-height: 1; } .slide-1 .main-title .subtitle { font-size: 1.05rem; letter-spacing: .06em; }
  .slide-1 .title-pill { padding: .8rem 2.2rem; margin-bottom: 1.6rem; font-size: .95rem; }
  .toc { display: flex; flex-direction: column; gap: .8rem; }
  .toc-row { display: flex; align-items: center; gap: 1.4rem; background: var(--card); border: 2px solid var(--outline); border-radius: 9999px;
             padding: .55rem 1.6rem .55rem .7rem; box-shadow: 5px 5px 0 var(--shadow); }
  .toc-row .card-icon { width: 3.2rem; height: 3.2rem; border-radius: 50%; border: 2px solid var(--outline); display: flex; align-items: center;
             justify-content: center; font-family: var(--font-display); font-size: 1.6rem; font-weight: 700; flex-shrink: 0; }
  .toc-t { flex: 1; } .toc-t h3 { font-family: var(--font-display); font-size: 1.45rem; line-height: 1.15; } .toc-t p { font-size: .95rem; opacity: .84; }
  .toc-row .pill { padding: .4rem 1.1rem; font-size: .95rem; font-weight: 600; }
  .slide-2 .left-content h2, .slide-9 .text-side h2 { font-size: 2.6rem; }
  .slide-2 .left-content p, .slide-9 .text-side p { font-size: 1.15rem; line-height: 1.5; opacity: .92; }
  .slide-2 .right-visual { height: 20rem; } .slide-2 .right-visual .orbit-center { width: 10rem; height: 10rem; font-size: 2.1rem; }
  .slide-3 .pillar-card { padding: 1.8rem 1.6rem; } .slide-3 .pillar-card p { font-size: 1.1rem; line-height: 1.45; opacity: .92; }
  .slide-3 .pillar-card .card-icon { width: 3.2rem; height: 3.2rem; }
  .slide-8 .diagram-node { font-size: 1.05rem; padding: .9rem 1.2rem; min-width: 0; white-space: nowrap; } .slide-8 .diagram-connector { width: 1.9rem; flex-shrink: 0; margin-right: 6px; }
  .notes-row { display: flex; justify-content: center; gap: 2rem; } .notes-row > div { flex: 1; max-width: 17rem; text-align: center; }
  .notes-row .pill { padding: .35rem 1.1rem; font-size: .95rem; font-weight: 600; margin-bottom: .5rem; } .notes-row p { font-size: 1.05rem; line-height: 1.35; }
  .slide-8 .notes-row { margin-top: 1.6rem; }
  .slide-6 .timeline-step { max-width: 12rem; } .slide-6 .timeline-step .step-node { width: 3.4rem; height: 3.4rem; font-size: 1.4rem; }
  .slide-6 .timeline-step .step-label { font-size: .98rem; text-transform: none; letter-spacing: 0; } .slide-6 .timeline-step .step-desc { font-size: 1rem; opacity: .9; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 1.6rem; }
  .panel { background: var(--card); border: 2px solid var(--outline); border-radius: 1.6rem; padding: 1.1rem 1.4rem; box-shadow: 6px 6px 0 var(--shadow); }
  .panel h3 { font-family: var(--font-display); font-size: 1.4rem; margin-bottom: .5rem; } .panel p { font-size: 1.05rem; line-height: 1.4; margin-top: .9rem; }
  .panel svg { width: 100%; display: block; } .panel img { width: 100%; display: block; border-radius: .6rem; }
  svg text { font-family: var(--font-body); fill: var(--fg); } svg .ax { font-size: 14px; opacity: .85; } svg .lb { font-size: 16px; }
  svg .halo { paint-order: stroke; stroke: #FCFBF6; stroke-width: 5px; }
  .chartslide .slide-inner.x-wide { gap: 1.1rem; } .take { text-align: center; font-size: 1.12rem; font-weight: 500; }
  .tbl { width: 100%; border-collapse: collapse; font-size: 1rem; } .tbl th, .tbl td { padding: .38rem .6rem; text-align: center; border-bottom: 1px solid #d8d6cc; }
  .tbl th:first-child { text-align: left; } .tbl tr:last-child th, .tbl tr:last-child td { border-bottom: 0; } .tbl td { font-variant-numeric: tabular-nums; }
  .tbl.big { font-size: 1.12rem; } .tbl.heat { font-size: 1.08rem; } .heat .rowh { font-family: var(--font-display); font-size: 1.3rem; }
  .legend { display: flex; justify-content: center; flex-wrap: wrap; gap: .3rem 1.4rem; font-size: .95rem; } .legend span { display: inline-flex; align-items: center; gap: .4rem; }
  .legend svg { width: 2.4rem; }
  .slide-9 .slide-inner.x-map { grid-template-columns: 1.45fr 1fr; max-width: 66rem; gap: 2rem; } .slide-9 .slide-inner.x-demo { grid-template-columns: 1.15fr 1fr; max-width: 64rem; gap: 2.2rem; }
  .x-map .panel { padding: .8rem; } .x-map .text-side h2 { font-size: 2.2rem; } .x-map .text-side p { font-size: 1.08rem; }
  .slide-9 .visual-side { height: 22rem; } .slide-9 .text-side .text-pills .mini-pill { font-size: .9rem; text-transform: none; }
  .stack { position: absolute; inset: 1.4rem; display: flex; flex-direction: column; justify-content: space-around; align-items: center; } .stack .pill { padding: .7rem; font-size: 1.05rem; font-weight: 600; }
  .src { font-size: .85rem; text-align: center; opacity: .85; } .slide-3 .slide-inner { gap: 1.8rem; }
  .slide-10 h2 { font-size: 4.2rem; } .slide-10 .closing-pill { font-size: 1rem; }
  @media print {
    @page { size: 13.333in 7.5in; margin: 0; }
    html { font-size: 18px; } html, body { height: auto; overflow: visible; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .grain-overlay, .nav-dots, .slide-counter { display: none; }
    .presentation { height: auto; }
    .slide { position: relative; inset: auto; width: 13.333in; height: 7.5in; opacity: 1; break-after: page; transition: none; }
    .slide.active > * { animation: none; }
    .slide:last-child { break-after: auto; }
  }
"""
n = len(slides)
dots = "".join(f'<div class="nav-dot{" active" if i == 1 else ""}" data-slide="{i}"></div>' for i in range(1, n + 1))
html = (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>Predictive maintenance from audio</title>\n<!-- Template: Capsule, MIT, Copyright (c) 2026 Zara Zhang (assets/TEMPLATE_LICENSE) -->\n"
        f"<style>{fonts}{style}{EXTRA}</style></head><body><div class=\"grain-overlay\"></div>\n<div class=\"presentation\">\n"
        + "\n".join(slides) + f'\n</div><div class="nav-dots">{dots}</div><div class="slide-counter"><span id="current">01</span> / {n:02d}</div>'
        f'<div class="nav-hint">Use arrow keys to navigate</div><script>{script}\n'
        "const h = parseInt(location.hash.slice(1)); if (h) goToSlide(h);</script></body></html>")
(P / "index.html").write_text(html, encoding="utf-8")
print(n, "slides", len(html) // 1024, "KB")
