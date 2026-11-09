"""MedScribe Classifier: Gradio app.

    python app.py            # http://127.0.0.1:7860
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.predict import highlight, load_model, predict

ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "results" / "results.json").read_text())
EDA = json.loads((ROOT / "results" / "eda.json").read_text())
EXAMPLES = {p.stem.replace("_", " "): p.read_text(encoding="utf-8") for p in sorted((ROOT / "examples").glob("*.txt"))}

ACCENT = "#0F766E"
MUTED = "#64748b"
PALETTE = ["#0F766E", "#0EA5E9", "#F59E0B", "#E11D48", "#8B5CF6", "#10B981", "#F97316",
           "#6366F1", "#84CC16", "#EC4899", "#14B8A6", "#A16207", "#64748B"]


# ---------------------------------------------------------------- styling
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c50="#f0fdfa", c100="#ccfbf1", c200="#99f6e4", c300="#5eead4", c400="#2dd4bf",
                                c500="#14b8a6", c600="#0d9488", c700="#0f766e", c800="#115e59", c900="#134e4a",
                                c950="#042f2e"),
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size="lg",
).set(button_primary_background_fill="#0F766E", button_primary_background_fill_hover="#115e59",
      button_primary_background_fill_dark="#0d9488", block_shadow="0 1px 3px rgba(15,23,42,.08)")

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero {padding: 22px 26px; border-radius: 18px; color: #fff;
       background: linear-gradient(135deg, #0F766E 0%, #115e59 55%, #134e4a 100%);}
#hero h1 {margin: 0 0 6px; font-size: 1.9rem; font-weight: 700; color: #fff;}
#hero p {margin: 0; opacity: .92; font-size: 1.02rem; color: #fff;}
#hero .pills {margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;}
#hero .pill {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
             padding: 3px 10px; border-radius: 999px; font-size: .82rem;}
#hero .warn {background: #FEF3C7; color: #78350F; border: none; font-weight: 600;}
.result-card {padding: 16px 18px; border-radius: 14px; border: 1px solid var(--border-color-primary);
              background: var(--block-background-fill);}
.result-card .big {font-size: 1.55rem; font-weight: 700; color: var(--color-accent);}
.result-card .sub {color: var(--body-text-color-subdued); font-size: .92rem;}
.kw {display:inline-block; margin: 3px 4px 0 0; padding: 2px 9px; border-radius: 999px; font-size: .84rem;
     background: rgba(15,118,110,.12); color: var(--body-text-color); border: 1px solid rgba(15,118,110,.35);}
.kw.neg {background: rgba(225,29,72,.10); border-color: rgba(225,29,72,.35);}
footer {display: none !important;}
"""

HERO = """
<div id="hero">
  <h1>🩺 MedScribe Classifier</h1>
  <p>Predicts the medical specialty of a clinical transcription and shows which words drove the decision.</p>
  <div class="pills">
    <span class="pill">13 specialties · MTSamples</span>
    <span class="pill">TF-IDF + Logistic Regression</span>
    <span class="pill">vs. BiLSTM · BiGRU · DistilBERT</span>
    <span class="pill warn">⚠ Research demo, not for clinical use</span>
  </div>
</div>
"""


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=MUTED),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(100,116,139,.2)", zerolinecolor="rgba(100,116,139,.3)")
    fig.update_yaxes(gridcolor="rgba(100,116,139,.2)", zerolinecolor="rgba(100,116,139,.3)")
    return fig


# ---------------------------------------------------------------- classify tab
def empty_prob_fig() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="Run a prediction to see class probabilities", showarrow=False,
                       font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style(fig, 380)


def classify(text: str):
    text = (text or "").strip()
    if len(text.split()) < 5:
        raise gr.Error("Please paste a transcription (at least a few sentences), or pick a sample.")
    r = predict(text)
    if r["n_known_terms"] == 0:
        raise gr.Error("None of the words in this text appear in the training vocabulary. Is it a clinical note?")

    probs = pd.Series(r["probs"]).head(8)[::-1]
    colors = [ACCENT if lab == r["label"] else "#94a3b8" for lab in probs.index]
    fig = go.Figure(go.Bar(x=probs.values, y=probs.index, orientation="h", marker_color=colors,
                           text=[f"{p:.0%}" for p in probs.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, min(1.0, probs.max() * 1.25 + 0.05)], tickformat=".0%")
    fig.update_layout(title="Top specialties by probability")
    fig = style(fig, 380)

    runner_up = list(r["probs"].items())[1]
    hedge = "" if r["confidence"] >= 0.5 else " · low confidence, treat as a guess"
    kws = "".join(f'<span class="kw">{w}</span>' for w, _ in r["keywords"][:10]) or "<i>none</i>"
    neg = "".join(f'<span class="kw neg">{w}</span>' for w, _ in r["against"][:5]) or "<i>none</i>"
    card = f"""
<div class="result-card">
  <div class="sub">Predicted specialty</div>
  <div class="big">{r['label']}</div>
  <div class="sub">{r['confidence']:.0%} confidence · runner-up: {runner_up[0]} ({runner_up[1]:.0%}){hedge}</div>
  <div style="margin-top:12px" class="sub">Words pushing <b>towards</b> {r['label']}:</div><div>{kws}</div>
  <div style="margin-top:8px" class="sub">Words pushing <b>against</b> it:</div><div>{neg}</div>
</div>"""
    return card, fig, highlight(text, r["keywords"])


def pick_example(name: str) -> str:
    return EXAMPLES.get(name, "")


# ---------------------------------------------------------------- EDA tab
def class_dist_fig() -> go.Figure:
    raw = pd.Series(EDA["raw_counts"]).sort_values()
    clean = pd.Series(EDA["clean_counts"])
    kept = raw.index.isin(clean.index)
    fig = go.Figure(go.Bar(x=raw.values, y=raw.index, orientation="h",
                           marker_color=[ACCENT if k else "#cbd5e1" for k in kept],
                           hovertemplate="%{y}: %{x} rows<extra></extra>"))
    fig.update_layout(title="Raw labels: 40 (teal = kept as a class)")
    return style(fig, 760)


def clean_dist_fig() -> go.Figure:
    s = pd.Series(EDA["clean_counts"]).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=ACCENT,
                           text=s.values, textposition="outside"))
    fig.update_layout(title=f"Clean dataset: {s.sum()} unique notes, {len(s)} specialties")
    return style(fig, 460)


def top_terms_fig(spec: str) -> go.Figure:
    t = pd.DataFrame(EDA["top_terms"][spec], columns=["term", "score"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["score"], y=t["term"], orientation="h", marker_color=ACCENT))
    fig.update_layout(title=f"Highest mean TF-IDF terms: {spec} (training split)")
    return style(fig, 460)


def tsne_fig() -> go.Figure:
    d = pd.DataFrame(EDA["tsne"])
    fig = px.scatter(d, x="x", y="y", color="label", hover_name="title", hover_data={"split": True, "x": False,
                     "y": False, "label": True}, color_discrete_sequence=PALETTE, opacity=0.8,
                     render_mode="svg")  # SVG, not WebGL, so it renders everywhere
    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(title="t-SNE of every note (TF-IDF → 50-d SVD → 2-D)", legend_title_text="")
    return style(fig, 620)


# ---------------------------------------------------------------- model tab
ORDER = sorted(RESULTS, key=lambda k: -RESULTS[k]["test"]["macro_f1"])
CLASSES = RESULTS[ORDER[0]]["classes"]


def results_table() -> pd.DataFrame:
    rows = []
    for k in ORDER:
        v = RESULTS[k]
        rows.append({"Model": k, "Test macro-F1": round(v["test"]["macro_f1"], 3),
                     "Test accuracy": round(v["test"]["accuracy"], 3),
                     "Val macro-F1": round(v["val"]["macro_f1"], 3),
                     "Train time (CPU)": f"{v.get('train_seconds', 0):.0f}s"})
    return pd.DataFrame(rows)


def compare_fig() -> go.Figure:
    t = results_table().iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=t["Model"], x=t["Test macro-F1"], name="Macro-F1", orientation="h", marker_color=ACCENT)
    fig.add_bar(y=t["Model"], x=t["Test accuracy"], name="Accuracy", orientation="h", marker_color="#94a3b8")
    fig.update_layout(barmode="group", title="Held-out test set", legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(range=[0, 1])
    return style(fig, 420)


def confusion_fig(model: str) -> go.Figure:
    cm = pd.DataFrame(RESULTS[model]["test"]["confusion"], index=CLASSES, columns=CLASSES)
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig = go.Figure(go.Heatmap(z=norm.values, x=CLASSES, y=CLASSES, colorscale=[[0, "rgba(15,118,110,0.03)"],
                    [1, ACCENT]], text=cm.values, texttemplate="%{text}", zmin=0, zmax=1,
                    hovertemplate="true %{y}<br>pred %{x}<br>%{text} notes<extra></extra>", showscale=False))
    fig.update_yaxes(autorange="reversed", title="True")
    fig.update_xaxes(title="Predicted", tickangle=40)
    fig.update_layout(title=f"Confusion matrix (test, row-normalised colour): {model}")
    return style(fig, 620)


def per_class_fig(model: str) -> go.Figure:
    pc = RESULTS[model]["test"]["per_class"]
    s = pd.Series({c: pc[c]["f1-score"] for c in CLASSES}).sort_values()
    n = {c: int(pc[c]["support"]) for c in CLASSES}
    fig = go.Figure(go.Bar(x=s.values, y=[f"{c} (n={n[c]})" for c in s.index], orientation="h",
                           marker_color=ACCENT, text=[f"{v:.2f}" for v in s.values], textposition="outside"))
    fig.update_xaxes(range=[0, 1.1])
    fig.update_layout(title="Per-class F1 on test")
    return style(fig, 520)


ABOUT = f"""
### How it works
1. **Cleaning.** MTSamples has {EDA['stats']['raw_rows']:,} rows but only {EDA['stats']['raw_unique_texts']:,} unique
   texts: most notes are filed twice, once under a specialty and once under a document type such as *Surgery* or
   *Consult - History and Phy.* We drop the document-type buckets, keep **one row per unique note**, drop notes still
   tagged with two specialties, and keep specialties with ≥ 40 notes → **{EDA['stats']['clean_rows']:,} notes,
   {EDA['stats']['clean_classes']} classes**. This avoids the duplicate leakage that inflates scores in many MTSamples
   tutorials.
2. **Split.** Stratified 70 / 15 / 15 train / val / test ({EDA['split_sizes']['train']} / {EDA['split_sizes']['val']} /
   {EDA['split_sizes']['test']}). Hyper-parameters picked on val; test scored once.
3. **Model.** Word + bigram TF-IDF → class-balanced logistic regression. Each highlighted word's weight is
   *tf-idf(word) × coefficient(word, predicted class)*, i.e. exactly what it adds to that class's score.
4. **Baselines.** Linear SVM, Complement NB, BiLSTM, BiGRU and a CPU DistilBERT fine-tune: see *Model comparison*.

**Limitations.** ~1.5k notes is small; several specialties overlap clinically (e.g. Neurology vs. Pain Management,
General Medicine vs. everything), and the model only sees words, not clinical context. Research demo only.

Originally developed as a team university project; this version was refactored and extended by Amna Choudhri.
"""


# ---------------------------------------------------------------- layout
with gr.Blocks(title="MedScribe Classifier") as demo:
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("🔎 Classify"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    sample = gr.Dropdown(list(EXAMPLES), label="Pick a sample transcription (held-out test notes)",
                                         value=None)
                    text = gr.Textbox(label="Transcription", lines=14, max_lines=22,
                                      placeholder="Paste a clinical note: HPI, exam, assessment, plan…")
                    with gr.Row():
                        clear = gr.ClearButton(value="Clear")
                        btn = gr.Button("Classify", variant="primary", scale=2)
                with gr.Column(scale=5):
                    card = gr.HTML('<div class="result-card"><div class="sub">Predicted specialty</div>'
                                   '<div class="big">-</div><div class="sub">Pick a sample or paste a note, '
                                   'then press <b>Classify</b>.</div></div>')
                    probs = gr.Plot(value=empty_prob_fig(), show_label=False)
            hl = gr.HighlightedText(label="Words that drove the prediction", combine_adjacent=False,
                                    color_map={"strong": "#0F766E", "supporting": "#5EEAD4"}, show_legend=True)
            sample.change(pick_example, sample, text)
            btn.click(classify, text, [card, probs, hl])
            text.submit(classify, text, [card, probs, hl])
            clear.add([text, hl])

        with gr.Tab("📊 Explore data"):
            with gr.Row():
                gr.Plot(class_dist_fig(), show_label=False)
                with gr.Column():
                    gr.Plot(clean_dist_fig(), show_label=False)
                    spec = gr.Dropdown(CLASSES, value=CLASSES[0], label="Top terms for specialty")
                    terms_plot = gr.Plot(top_terms_fig(CLASSES[0]), show_label=False)
            spec.change(top_terms_fig, spec, terms_plot)
            gr.Plot(tsne_fig(), show_label=False)

        with gr.Tab("🏁 Model comparison"):
            gr.Markdown("All numbers are on the same **held-out test set** "
                        f"({EDA['split_sizes']['test']} notes, never used for training or model selection).")
            gr.Dataframe(results_table(), interactive=False, wrap=True)
            gr.Plot(compare_fig(), show_label=False)
            pick = gr.Dropdown(ORDER, value=ORDER[0], label="Inspect model")
            cm_plot = gr.Plot(confusion_fig(ORDER[0]), show_label=False)
            pc_plot = gr.Plot(per_class_fig(ORDER[0]), show_label=False)
            pick.change(lambda m: (confusion_fig(m), per_class_fig(m)), pick, [cm_plot, pc_plot])

        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)

load_model()  # fail fast with a clear message if the model is missing

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
