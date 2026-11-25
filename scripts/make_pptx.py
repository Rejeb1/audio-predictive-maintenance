"""Editable PowerPoint version of the deck (python-pptx): presentation/presentation.pptx

Native text, shapes and tables, no slide screenshots. Charts are drawn with shapes because chart
objects are not reliably kept by Canva. Fonts exist in Canva (Bodoni Moda, Space Grotesk).
"""
import csv
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE, MSO_THEME_COLOR as T
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE as M
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
R, P = ROOT / "results", ROOT / "presentation"
rows = lambda f: list(csv.DictReader(open(R / f, encoding="utf-8")))
S = {(r["dataset"], r["machine"], r["model"]): {k: 100 * float(r[k]) for k in ("auc", "ci_low", "ci_high")} for r in rows("detection_summary.csv")}
ID = {(r["machine"], r["id"], r["model"]): 100 * float(r["auc"]) for r in rows("detection_per_id.csv") if r["dataset"] == "dcase"}
DEC = {(r["machine"], r["model"]): r for r in rows("decision_dcase.csv") if r["percentile"] == "90.0"}
Q = {(r["machine"], r["model"], r["variant"]): r for r in rows("edge_quality.csv")}
LAT = {r["stage"]: float(r["median_ms"]) for r in rows("edge_latency.csv") if r["threads"] == "1"}

HEAD, BODY = "Bodoni Moda", "Space Grotesk"
COL = {"coral": T.ACCENT_1, "lime": T.ACCENT_2, "sky": T.ACCENT_3, "violet": T.ACCENT_4, "yellow": T.ACCENT_5,
       "lavender": T.ACCENT_6, "ink": T.TEXT_1, "bg": T.BACKGROUND_1, "card": T.BACKGROUND_2, "blue": T.TEXT_2}
THEME = {"dk1": "1A1A1A", "lt1": "F5F5F0", "dk2": "2D5FBF", "lt2": "FCFBF6", "accent1": "E85D4E", "accent2": "C4D94E",
         "accent3": "8BB4F7", "accent4": "A06CE8", "accent5": "F2D160", "accent6": "C5B5E0", "hlink": "2D5FBF", "folHlink": "7A3FD1"}
TEAM = [("Mohamed Aziz Ben Rejeb", "coral"), ("Wael Znagui", "lime"), ("Maram Nuigues", "sky"),
        ("Khalil Znazen", "violet"), ("Aymen Turki", "yellow")]
PARTS = ["Context and data", "Autoencoders", "Pretrained embeddings", "Results and robustness", "Explainability and deployment"]
MODELS = [("dense_ae", "Dense AE"), ("conv_ae", "Conv AE"), ("panns_knn_type", "PANNs kNN, type"),
          ("panns_gmm_type", "PANNs GMM, type"), ("panns_knn_id", "PANNs kNN, ID"), ("panns_gmm_id", "PANNs GMM, ID")]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
prs.core_properties.title = "Hearing faults in machine noise"
prs.core_properties.author = ", ".join(n for n, _ in TEAM)

# theme: fonts and colours in one place
master = prs.slide_master
tp = next(r.target_part for r in master.part.rels.values() if r.reltype.endswith("/theme"))
x = etree.fromstring(tp.blob)
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
for k, v in THEME.items():
    e = x.find(f".//{A}clrScheme/{A}{k}")
    for c in list(e):
        e.remove(c)
    etree.SubElement(e, f"{A}srgbClr").set("val", v)
for k, f in (("majorFont", HEAD), ("minorFont", BODY)):
    x.find(f".//{A}{k}/{A}latin").set("typeface", f)
tp._blob = etree.tostring(x, xml_declaration=True, encoding="UTF-8", standalone=True)

# one layout: title + free area + slide number
layout = prs.slide_layouts[5]
for lay in list(prs.slide_layouts):
    if lay is not layout:
        prs.slide_layouts.remove(lay)
layout.name = "Title and free area"
bg = master.background.fill
bg.solid()
bg.fore_color.theme_color = T.BACKGROUND_1
for holder in (master, layout):
    for sh in list(holder.placeholders):
        if sh.placeholder_format.type == 1 or sh.placeholder_format.idx == 0:
            sh.left, sh.top, sh.width, sh.height = Inches(.6), Inches(.42), Inches(12.13), Inches(.85)
        elif holder is layout:
            sh._element.getparent().remove(sh._element)
        else:
            sh.left, sh.width = Inches(.6), Inches(12.13)
            if sh.placeholder_format.idx in (10, 11, 12):  # date, footer, number: one row at the bottom
                sh.width, sh.top, sh.height = Inches(3.5), Inches(6.9), Inches(.4)
                sh.left = Inches({10: .6, 11: 4.9, 12: 9.2}[sh.placeholder_format.idx])
st = master._element.find(f".//{qn('p:titleStyle')}/{qn('a:lvl1pPr')}")
st.find(qn("a:defRPr")).set("sz", "3600")
st.find(qn("a:defRPr")).set("b", "1")


def number(shapes):
    tb = shapes.add_textbox(Inches(11.9), Inches(6.93), Inches(.85), Inches(.35)) if hasattr(shapes, "add_textbox") else None
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    f = etree.SubElement(p._p, qn("a:fld"))
    f.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    f.set("type", "slidenum")
    etree.SubElement(f, qn("a:rPr")).set("sz", "1200")
    etree.SubElement(f, qn("a:t")).text = "0"


def color(fmt, c):
    if c in COL:
        fmt.theme_color = COL[c]
    else:
        fmt.rgb = RGBColor.from_string(c)


def fill(sh, c, line=True, w=1.5):
    if c is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        color(sh.fill.fore_color, c)
    if line:
        color(sh.line.color, "ink")
        sh.line.width = Pt(w)
    else:
        sh.line.fill.background()
    for e in sh._element.findall(qn("p:style")):
        sh._element.remove(e)  # no inherited effects
    return sh


def put(tf, text, size=16, bold=False, align="l", font=None, c="ink", anchor="m", margin=0.0, gap=0):
    """text: string or list of paragraphs; **bold** runs are supported."""
    tf.word_wrap = True
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = 0
    for i, t in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if i and gap:
            p.space_before = Pt(gap)
        for k, part in enumerate(re.split(r"\*\*", t)):
            if not part:
                continue
            for j, piece in enumerate(part.split("\n")):
                if j:
                    p.add_line_break()
                r = p.add_run()
                r.text = piece
                r.font.size, r.font.bold = Pt(size), bold or k % 2 == 1
                color(r.font.color, c)
                if font:
                    r.font.name = font


def text(s, x, y, w, h, t, size=16, **k):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    put(tb.text_frame, t, size, **k)
    return tb


def box(s, kind, x, y, w, h, c, t=None, size=16, radius=None, shadow=0.0, rot=0, line=True, name=None, **k):
    if shadow:  # hard offset shadow of the Capsule style
        b = fill(s.shapes.add_shape(kind, Inches(x + shadow), Inches(y + shadow), Inches(w), Inches(h)), "ink")
        b.name = "Shadow"
        b.rotation = rot
        if radius is not None:
            b.adjustments[0] = min(.5, radius / min(w, h))
    sh = fill(s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h)), c, line)
    if radius is not None:
        sh.adjustments[0] = min(.5, radius / min(w, h))
    sh.rotation = rot
    if name:
        sh.name = name
    if t is not None:
        put(sh.text_frame, t, size, **{"align": "c", "margin": .08, **k})
    return sh


def pill(s, x, y, w, h, c, t, size=14, **k):
    return box(s, M.ROUNDED_RECTANGLE, x, y, w, h, c, t, size, radius=9, **{"bold": True, **k})


def panel(s, x, y, w, h, c="card"):
    return box(s, M.ROUNDED_RECTANGLE, x, y, w, h, c, radius=.28, shadow=.08, name="Panel")


def line(s, x1, y1, x2, y2, c="ink", w=1.5, dash=None, arrow=False):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    for e in ln._element.findall(qn("p:style")):
        ln._element.remove(e)
    color(ln.line.color, c)
    ln.line.width = Pt(w)
    if dash:
        ln.line.dash_style = dash
    if arrow:
        etree.SubElement(ln.line._get_or_add_ln(), qn("a:tailEnd")).set("type", "triangle")
    return ln


NOTES = {}
for block in re.split(r"\n(?=\*\*Slide )", (P / "speaker_script.md").read_text(encoding="utf-8").split("\n---")[0]):
    m = re.match(r"\*\*Slide (\d+)[^*]*\*\*\s*(.*?)(?=\n##|\Z)", block, re.S)
    if m:
        NOTES[int(m[1])] = m[2].replace("*", "").strip()


def slide(title, part=None, size=None, at=None, align="c"):
    s = prs.slides.add_slide(layout)
    t = s.shapes.title
    if at:
        t.left, t.top, t.width, t.height = (Inches(v) for v in at)
    put(t.text_frame, title, size or 36, bold=True, align=align, font=None)
    n = len(prs.slides)
    if part is not None:
        w = .55 + .115 * len(PARTS[part])
        pill(s, .6, 6.9, w, .38, TEAM[part][1], f"{part + 1}. {PARTS[part]}", 12)
        text(s, .6 + w + .15, 6.9, 4, .38, TEAM[part][0], 14, bold=True)
    if n in NOTES:
        s.notes_slide.notes_text_frame.text = NOTES[n]
    return s


def sub(s, t):
    text(s, .6, 1.27, 12.13, .4, t, 16, align="c")


def flow(s, nodes, y=2.35, h=1.0, gap=.42):
    ws = [.62 + .118 * max(len(l) for l in t.split("\n")) for t, _ in nodes]
    k = min(1, (12.13 - gap * (len(nodes) - 1)) / sum(ws))
    ws = [w * k for w in ws]
    x = (13.333 - sum(ws) - gap * (len(nodes) - 1)) / 2
    for i, ((t, c), w) in enumerate(zip(nodes, ws)):
        pill(s, x, y, w, h, c, t, 15, shadow=.07, margin=.04)
        if i < len(nodes) - 1:
            line(s, x + w + .09, y + h / 2, x + w + gap - .02, y + h / 2, w=2, arrow=True)
        x += w + gap


def notes_row(s, items, y, w=3.4, size=16, h=1.0):
    gap = .4
    w = min(w, (12.13 - gap * (len(items) - 1)) / len(items))
    x = (13.333 - len(items) * w - gap * (len(items) - 1)) / 2
    for t, c, d in items:
        pw = .6 + .125 * len(t)
        pill(s, x + (w - pw) / 2, y, pw, .4, c, t, 14)
        text(s, x, y + .5, w, h, d, size, align="c", anchor="t")
        x += w + gap


def cards(s, items, y=1.75, h=4.6):
    w, gap = 3.75, .44
    x = (13.333 - 3 * w - 2 * gap) / 2
    for i, c, t, d in items:
        panel(s, x, y, w, h)
        box(s, M.OVAL, x + .35, y + .35, .75, .75, c, i, 20, bold=True, font=HEAD, margin=0)
        text(s, x + .35, y + 1.3, w - .7, .95, t, 22, bold=True, font=HEAD, anchor="t")
        text(s, x + .35, y + 2.25, w - .7, h - 2.4, d, 16, anchor="t")
        x += w + gap


def table(s, x, y, w, data, widths, size=15, row=.44, first_left=1, heat=None, bold_best=False):
    gf = s.shapes.add_table(len(data), len(data[0]), Inches(x), Inches(y), Inches(w), Inches(row * len(data)))
    tb = gf.table
    tp_ = gf._element.graphic.graphicData.tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tp_.set(a, "0")
    sid = tp_.find(qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # no style, no grid
    for j, f in enumerate(widths):
        tb.columns[j].width = Inches(w * f / sum(widths))
    for i, r in enumerate(data):
        tb.rows[i].height = Inches(row)
        nums = [v for v in r if isinstance(v, float)]
        for j, v in enumerate(r):
            c = tb.cell(i, j)
            isnum = isinstance(v, float)
            put(c.text_frame, f"{v:.1f}" if isnum else v, size, align="l" if j < first_left else "c",
                bold=i == 0 or j < first_left or (bold_best and isnum and v == max(nums)))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_top = c.margin_bottom = 0
            c.margin_left = c.margin_right = Inches(.08)
            if heat and isnum:
                k = max(0, min(1, (v - 50) / 45))
                rgb = [round(a + (b - a) * k) for a, b in zip((252, 251, 246), (139, 180, 247))]
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(*rgb)
            else:
                c.fill.background()
            if i < len(data) - 1:  # thin rule under each row
                pr = c._tc.get_or_add_tcPr()
                ln = etree.SubElement(pr, qn("a:lnB"))
                ln.set("w", "9525")
                sf = etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr"))
                sf.set("val", "D2D0C6")
                pr.insert(0, ln)  # borders come before the fill in the schema
    return tb


# the slide number lives on the layout
from pptx.shapes.shapetree import LayoutShapes  # noqa: E402

tmp = prs.slides.add_slide(layout)
number(tmp.shapes)
el = tmp.shapes[-1]._element
layout.shapes._spTree.append(el)
rid = prs.slides._sldIdLst[0]
prs.part.drop_rel(rid.rId)
prs.slides._sldIdLst.remove(rid)


def decor(s, items):
    for t, c, x, y, r in items:
        pill(s, x, y, .55 + .115 * len(t), .46, c, t, 13, rot=r)


def team(s, y):
    ws = [.5 + .118 * len(n) for n, _ in TEAM]
    x = (13.333 - sum(ws) - .2 * 4) / 2
    for (n, c), w in zip(TEAM, ws):
        pill(s, x, y, w, .5, c, n, 14, shadow=.06, margin=.04)
        x += w + .2


# 1 title, 2 contents
s = slide("Hearing faults\nin machine noise", size=60, at=(1.2, 1.75, 10.93, 2.1))
pill(s, 3.9, .85, 5.53, .55, "yellow", "Deep Learning mini-project · IIA5 · INSAT", 15)
text(s, 1.2, 3.95, 10.93, .5, "Predictive maintenance from audio: unsupervised detection of machine faults", 18, align="c")
team(s, 4.85)
text(s, 1.2, 5.65, 10.93, .4, "Supervisor: Ridha ALOUI · December 2026", 16, align="c")
decor(s, [("PUMPS", "coral", 1.0, .75, -12), ("LOG-MEL", "lavender", 11.0, .8, 8), ("NO FAULT LABELS", "sky", .5, 2.3, 6),
          ("AUC", "F5B895", 11.6, 2.5, -8), ("FANS", "lime", 6.1, 6.55, 4)])

s = slide("Table of contents")
for i, (p, d) in enumerate(zip(PARTS, ("The problem, the pipeline and the two datasets", "Learning to rebuild normal sound",
                                       "PANNs, test protocol and overall results", "Per machine, under noise, with an alert threshold",
                                       "Error maps, ONNX and INT8, demo, conclusion"))):
    y = 1.5 + i * 1.07
    box(s, M.ROUNDED_RECTANGLE, 1.2, y, 10.93, .88, "card", radius=9, shadow=.07, name="Row")
    box(s, M.OVAL, 1.33, y + .11, .66, .66, TEAM[i][1], str(i + 1), 20, bold=True, font=HEAD, margin=0)
    text(s, 2.25, y + .08, 6.2, .42, p, 20, bold=True, font=HEAD)
    text(s, 2.25, y + .48, 6.2, .32, d, 14)
    w = .5 + .118 * len(TEAM[i][0])
    pill(s, 11.85 - w, y + .2, w, .48, TEAM[i][1], TEAM[i][0], 14, margin=.04)

# part 1
s = slide("A failing machine often\nsounds different first", 0, at=(.6, 1.1, 6.4, 1.5), align="l")
text(s, .6, 2.9, 6.1, 3.4, ["Worn bearings, leaks and imbalance change a machine's acoustic signature. Today, spotting this relies on experienced technicians or costly vibration sensors.",
                            "**Question:** can we detect anomalies from plain audio, learning **only from normal sounds**, despite factory noise?"], 18, anchor="t", gap=16)
box(s, M.OVAL, 8.85, 2.55, 2.3, 2.3, "lime", "Normal", 28, bold=True, font=HEAD, margin=0)
for t, c, px, py, r in (("LEAK", "coral", 7.9, 1.6, -14), ("IMBALANCE", "lavender", 10.9, 1.5, 10), ("WORN BEARING", "yellow", 7.3, 5.2, 8), ("UNKNOWN FAULT", "sky", 10.6, 5.3, -8)):
    pill(s, px, py, .55 + .115 * len(t), .5, c, t, 13, rot=r, shadow=.06)
s = slide("From a sound to an alert", 0)
flow(s, [("Audio\n10 s, 16 kHz", "card"), ("Log-mel\nspectrogram", "lavender"), ("Model of\nnormal sound", "lime"), ("Anomaly\nscore", "sky"), ("Threshold", "yellow"), ("Verdict and\nerror map", "coral")])
notes_row(s, [("Learn", "lime", "Train on normal sounds only: no fault is ever shown to the model"), ("Compare", "sky", "Three approaches, two machine types, three noise levels"),
              ("Explain", "coral", "Show which frequencies and moments look abnormal")], 4.3)
s = slide("Two datasets, two questions", 0)
for px, t, data, wd, d in (
        (.6, "DCASE 2020 Task 2: main comparison", [["", "Train (normal)", "Test normal", "Test fault"], ["Pump", "3,349", "400", "456"], ["Fan", "3,675", "400", "1,475"]], (1, 1.6, 1.4, 1.3),
         "4 machines (IDs) per type. 10% of the training normals are held out to set thresholds."),
        (6.87, "MIMII subset: noise study", [["Per machine and noise level", "Clips"], ["Normal: train / held out / test", "135 / 15 / 50"], ["Faults (test only)", "50"]], (2, 1),
         "Same clips at −6, 0 and +6 dB. New models per level: DCASE comes from MIMII, so reusing them would leak data.")):
    panel(s, px, 1.75, 5.86, 4.4)
    text(s, px + .35, 2.0, 5.2, .5, t, 20, bold=True, font=HEAD)
    table(s, px + .3, 2.65, 5.26, data, wd, 14)
    text(s, px + .35, 4.3, 5.16, 1.6, d, 16, anchor="t")

# part 2
s = slide("Three ways to model normal sound", 1)
cards(s, [("I", "coral", "Dense autoencoder", "Rebuilds 5 short frames at a time. Same design as the official DCASE baseline."),
          ("II", "lime", "Convolutional autoencoder", "Treats the spectrogram as an image and rebuilds time-frequency patches."),
          ("III", "sky", "Pretrained embeddings", "PANNs describes each clip; we measure the distance to normal clips.")])
s = slide("Rebuild badly = suspicious", 1)
sub(s, "Trained on normal sounds, an autoencoder should fail to rebuild what it has never heard")
flow(s, [("Input\n640", "card"), ("Encoder\n4 x 128", "lavender"), ("Bottleneck\n8", "coral"), ("Decoder\n4 x 128", "lavender"), ("Output\n640", "card")], 2.5)
notes_row(s, [("Input", "card", "5 frames x 128 mel bands"), ("Training", "lime", "100 epochs, Adam 1e-3, one model per machine type"),
              ("Score", "yellow", "Mean squared reconstruction error over the clip")], 4.4)
s = slide("Convolutional\nautoencoder", 1, at=(7.2, 1.2, 5.5, 1.4), align="l")
box(s, M.ROUNDED_RECTANGLE, .8, 1.3, 5.7, 5.0, "sky", radius=.35, shadow=.1)
for i, (t, c, w) in enumerate((("Patch 128 x 64", "card", 5.0), ("4 strided convolutions", "lavender", 4.0), ("Bottleneck 40", "coral", 2.5), ("4 transposed convolutions", "lavender", 4.0), ("Rebuilt patch", "card", 5.0))):
    pill(s, .8 + (5.7 - w) / 2, 1.62 + i * .9, w, .64, c, t, 16)
text(s, 7.2, 2.85, 5.4, 2.4, ["Each mel band is standardised, then the network sees 128 x 64 patches: about 2 seconds of context instead of 5 frames.", "80 epochs, OneCycle learning rate up to 2e-3."], 18, anchor="t", gap=14)
for t, c, px, py, w in (("More context", "lime", 7.2, 5.3, 1.75), ("About 1.5 MB", "sky", 9.1, 5.3, 1.75), ("Trained on Kaggle T4", "yellow", 7.2, 5.85, 2.6)):
    pill(s, px, py, w, .42, c, t, 14)

# part 3
s = slide("PANNs: a network that already knows sound", 2)
flow(s, [("10 s clip", "card"), ("CNN14, pretrained\non AudioSet", "lavender"), ("2,048 numbers", "lime"), ("kNN or GMM", "sky"), ("Score", "coral")])
notes_row(s, [("kNN", "sky", "Mean distance to the 5 nearest normal clips"), ("GMM", "violet", "How unlikely the clip is under a model of normal clips"),
              ("Per type", "lime", "Reference: all 4 machines, like the autoencoders"), ("Per ID", "yellow", "Reference: that exact machine, so its ID must be known")], 4.3)
s = slide("How we kept the comparison honest", 2)
line(s, 1.6, 2.85, 11.75, 2.85, w=3)
for i, (t, c, d) in enumerate((("No tuning on test", "coral", "Settings fixed in advance, all variants reported"), ("Separate data", "lime", "Thresholds use held-out normal clips only"),
                               ("AUC and pAUC", "sky", "Official metrics, averaged over 4 machines"), ("95% intervals", "violet", "Bootstrap over test clips; one training run per model"),
                               ("Training checks", "yellow", "A faulty laptop GPU was caught; runs redone on Kaggle"))):
    px = .6 + i * 2.43
    box(s, M.OVAL, px + .8, 2.43, .84, .84, c, str(i + 1), 22, bold=True, font=HEAD, shadow=.06, margin=0)
    text(s, px, 3.6, 2.43, .45, t, 17, bold=True, align="c")
    text(s, px + .12, 4.1, 2.19, 1.7, d, 16, align="c", anchor="t")


def forest(s, m, px, off):
    panel(s, px, 1.8, 5.86, 4.2)
    text(s, px + .3, 1.95, 2, .4, m.capitalize(), 20, bold=True, font=HEAD)
    X = lambda v: px + 2.2 + (v - 45) / 42 * 3.3
    for v in (50, 60, 70, 80):
        line(s, X(v), 2.65, X(v), 5.5, "D8D6CC", 1)
        text(s, X(v) - .3, 5.55, .6, .3, str(v), 12, align="c")
    line(s, X(off), 2.65, X(off), 5.5, "ink", 1.5, MSO_LINE.DASH)
    text(s, X(off) - 1.3, 2.3, 2.6, .3, f"official baseline {off:.1f}", 12, align="c")
    best = max(MODELS, key=lambda k: S[("dcase", m, k[0])]["auc"])[0]
    for i, (k, lab) in enumerate(MODELS):
        r, y, c = S[("dcase", m, k)], 2.95 + i * .46, "coral" if k == best else "blue"
        text(s, px + .15, y - .17, 1.95, .34, lab, 13, align="r", bold=k == best)
        line(s, X(r["ci_low"]), y, X(r["ci_high"]), y, c, 3)
        box(s, M.OVAL, X(r["auc"]) - .09, y - .09, .18, .18, c, line=False)
        fill(text(s, X(r["ci_high"]) + .06, y - .14, .42, .28, f"{r['auc']:.1f}", 13, bold=k == best, align="c"), "card", line=False)  # hides the baseline rule


s = slide("No single winner on DCASE", 2)
sub(s, "Mean AUC (%) and 95% interval; 50 is chance. Best in coral")
forest(s, "pump", .6, 72.89)
forest(s, "fan", 6.87, 65.83)
text(s, .6, 6.2, 12.13, .6, "Our dense AE matches the official baseline. PANNs GMM per ID leads on pump, but its interval overlaps the dense AE.", 16, align="c", bold=True)

# part 4
s = slide("The machine matters as much as the method", 3)
sub(s, "DCASE AUC (%) per machine. Bold: best in the row")
panel(s, .6, 1.8, 12.13, 4.3)
data = [["", ""] + [l for _, l in MODELS]]
for m in ("pump", "fan"):
    data += [[m.capitalize(), i.replace("id_", "ID ")] + [ID[(m, i, k)] for k, _ in MODELS] for i in ("id_00", "id_02", "id_04", "id_06")]
table(s, .85, 2.0, 11.63, data, (1, 1, 1.3, 1.3, 1.9, 1.9, 1.7, 1.7), 14, .43, 2, heat=True, bold_best=True)
text(s, .6, 6.28, 12.13, .45, "Conv AE: 93.3 on pump ID 04, but 48.8 (chance) on ID 02. Fan ID 00 is hard for every method.", 16, align="c", bold=True)

LINES = [("dense_ae", "Dense AE", "blue", M.OVAL, None), ("conv_ae", "Conv AE", "coral", M.RECTANGLE, None), ("panns_knn_id", "PANNs kNN, ID", "violet", M.DIAMOND, None),
         ("panns_gmm_id", "PANNs GMM, ID", "ink", M.ISOSCELES_TRIANGLE, None), ("panns_knn_type", "PANNs kNN, type", "8D8C86", M.OVAL, MSO_LINE.DASH),
         ("panns_gmm_type", "PANNs GMM, type", "8D8C86", M.RECTANGLE, MSO_LINE.ROUND_DOT)]


def noise(s, m, px):
    panel(s, px, 1.8, 5.86, 3.85)
    text(s, px + .3, 1.9, 2, .4, m.capitalize(), 20, bold=True, font=HEAD)
    X = {-6: px + 1.1, 0: px + 3.15, 6: px + 5.2}
    Y = lambda v: 5.2 - (v - 47) / 53 * 2.8
    for v in (50, 60, 70, 80, 90, 100):
        line(s, px + .75, Y(v), px + 5.6, Y(v), "D8D6CC", 1)
        text(s, px + .1, Y(v) - .15, .55, .3, str(v), 12, align="r")
    for k, vx in X.items():
        text(s, vx - .6, 5.25, 1.2, .32, {-6: "−6 dB", 0: "0 dB", 6: "+6 dB"}[k], 13, align="c")
    for k, _, c, mk, dash in reversed(LINES):
        pts = [(X[d], Y(S[(f"mimii_{d}dB", m, k)]["auc"])) for d in X]
        for a, b in zip(pts, pts[1:]):
            line(s, *a, *b, c, 2.5, dash)
        for a, b in pts:
            box(s, mk, a - .09, b - .09, .18, .18, c, line=False)


s = slide("Noise is a major limit", 3)
sub(s, "MIMII subset: mean AUC (%) against signal-to-noise ratio")
noise(s, "pump", .6)
noise(s, "fan", 6.87)
px = .75
for _, lab, c, mk, dash in LINES:
    line(s, px, 6.0, px + .42, 6.0, c, 2.5, dash)
    box(s, mk, px + .12, 5.91, .18, .18, c, line=False)
    text(s, px + .5, 5.83, 1.8, .34, lab, 13)
    px += .72 + .098 * len(lab)
text(s, .6, 6.35, 12.13, .4, "From −6 to +6 dB, every method gains 8 to 30 points. The dense AE reaches 94.3 on fan at +6 dB.", 16, align="c", bold=True)

s = slide("With a real threshold, most faults are missed", 3)
sub(s, "DCASE, threshold at the 90th percentile of held-out normal scores")
panel(s, .6, 1.8, 12.13, 3.35)
pc = lambda v: f"{100 * float(v):.0f}%"
data = [["", "Method", "Precision", "Recall", "F1", "False alarms", "Balanced accuracy"]]
for m in ("pump", "fan"):
    for k, lab in (MODELS[0], MODELS[1], MODELS[5]):
        d = DEC[(m, k)]
        data.append([m.capitalize(), lab, pc(d["precision"]), pc(d["recall"]), f"{float(d['f1']):.2f}", pc(d["false_alarm_rate"]), pc(d["balanced_acc"])])
table(s, .85, 1.95, 11.63, data, (1, 2, 1.4, 1.1, 1, 1.7, 2.2), 15, .435, 2)
notes_row(s, [("Missed faults", "coral", "Dense AE misses 59% on pump and 72% on fan"), ("F1 can mislead", "yellow", "Alarming on every clip scores 0.69 on pump and 0.88 on fan, yet is useless"),
              ("Target missed", "lavender", "False alarms often exceed the 10% target")], 5.4, 3.8, 15, .9)

# part 5
s = slide("Error maps\nshow where", 4, at=(8.75, .75, 4.0, 1.5), align="l")
panel(s, .6, .7, 7.8, 5.85)
pic = s.shapes.add_picture(str(R / "figures" / "error_map_pump_en.png"), Inches(.75), Inches(.85), width=Inches(7.5))
pic._element.nvPicPr.cNvPr.set("descr", "Spectrograms and error maps: a normal pump clip, the fault with the highest score and the fault with the lowest score")
text(s, 8.75, 2.4, 4.0, 4.2, ["**Middle:** the error follows the moving tones of this pump fault.", "**Right:** this fault sounds normal to the model, so it is missed.",
                             "**Caution:** a high score means unusual, not necessarily broken. On one fan clip, short tones that may be factory noise raise the score."], 16, anchor="t", gap=12)
s = slide("Small enough for the edge", 4)
sub(s, "ONNX Runtime, laptop i5-10300H, 1 thread, median per 10 s clip; not tested on an embedded board")
panel(s, .6, 1.8, 12.13, 3.0)
data = [["", "Size", "Pump AUC", "Fan AUC", "Model time"]]
for k, v, lab in (("dense", "fp32", "Dense FP32"), ("dense", "int8_dynamic", "Dense INT8 dynamic"), ("dense", "int8_static", "Dense INT8 static"), ("conv", "fp32", "Conv FP32"), ("conv", "int8_static", "Conv INT8 static")):
    data.append([lab, f"{float(Q[('pump', k, v)]['size_kb']):,.0f} KB", f"{100 * float(Q[('pump', k, v)]['auc']):.1f}", f"{100 * float(Q[('fan', k, v)]['auc']):.1f}", f"{LAT[k + '_' + v]:.1f} ms"])
table(s, .85, 1.97, 11.63, data, (2, 1, 1, 1, 1), 16, .44)
notes_row(s, [("3.5x smaller", "lime", "Dense INT8 dynamic keeps its AUC"), ("About 8 ms", "sky", "Whole dense INT8 pipeline, log-mel included"),
              ("820 ms", "coral", "PANNs CNN14 in PyTorch: about 100 times slower, 80M parameters")], 5.1, 3.8, 16)
s = slide("Live demo", 4, at=(7.7, 1.3, 5.0, .9), align="l")
panel(s, .6, .85, 6.6, 5.75)
pic = s.shapes.add_picture(str(P / "assets" / "demo.jpg"), Inches(.8), Inches(1.0), height=Inches(5.45))
pic.left = Inches(.6 + 3.3) - pic.width // 2
if pic.width > Inches(6.2):
    k = Inches(6.2) / pic.width
    pic.width, pic.height = Emu(int(pic.width * k)), Emu(int(pic.height * k))
    pic.left, pic.top = Inches(.8), Inches(.85 + 5.75 / 2) - pic.height // 2
pic._element.nvPicPr.cNvPr.set("descr", "Demo page showing an unusual pump sound")
text(s, 7.7, 2.4, 5.0, 2.6, ["Upload or record a sound, choose the machine, and get a plain verdict, the score among healthy clips, and the error over the spectrogram.",
                             "Each machine has a healthy clip, a typical fault and a clear fault. The typical fault is **missed on 7 of 8 machines**."], 18, anchor="t", gap=14)
for t, c, px, py, w in (("FastAPI + Gradio", "lime", 7.7, 5.2, 2.2), ("ONNX on CPU", "sky", 10.1, 5.2, 1.8), ("localhost:7860", "yellow", 7.7, 5.8, 2.0)):
    pill(s, px, py, w, .42, c, t, 14)
s = slide("What we learned, and what is missing", 4)
cards(s, [("1", "lime", "Findings", "Normal sounds alone can reveal faults. The best method depends on the machine, and noise is a major limit. The larger models did not always win."),
          ("2", "coral", "Limits", "Most faults are missed at our threshold. Two machine types, one training run per model, and no other microphone or factory tested."),
          ("3", "sky", "Next steps", "One autoencoder per machine, scoring the most abnormal frames, self-supervision with machine IDs, and recalibration on site.")], 1.5, 4.55)
text(s, .6, 6.25, 12.13, .5, "Data: DCASE 2020 Task 2 (Koizumi et al., Zenodo 3678171, CC BY-NC-SA 4.0); MIMII (Purohit et al., 2019, Zenodo 3384388, CC BY-SA 4.0). Model: PANNs (Kong et al., 2020).", 11, align="c")
s = slide("Thank you for listening", size=54, at=(1.2, 2.6, 10.93, 1.3))
pill(s, 5.5, 1.5, 2.33, .6, "yellow", "Questions?", 17)
team(s, 4.6)
decor(s, [("PUMPS", "lime", 1.4, 1.1, -12), ("FANS", "sky", 11.0, 1.2, 12), ("NOISE", "lavender", 1.6, 6.0, 10), ("DEMO", "coral", 10.8, 5.9, -8)])

prs.save(P / "presentation.pptx")
print(len(prs.slides), "slides", sorted(set(range(1, 20)) - set(NOTES)))
