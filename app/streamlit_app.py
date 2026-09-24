"""Dashboard for the accelerometer-only sleep staging project.

    streamlit run app/streamlit_app.py

Written for someone meeting the project for the first time. Each section opens
with a plain-English explanation; each chart says how to read it and closes
with what it shows; the technical detail lives in expanders and the Glossary
tab for anyone who wants it.

Every takeaway sentence is computed from the saved results rather than typed
in, so it stays true when the experiments are rerun. That rule is not
decoration: a hand-written caption once claimed the model's mistakes bunch up
at stage changes, and measured, they mostly do not.

Reads only what the training runs wrote to ``results/`` and the epoch cache.
It never trains and never imports torch, so it opens instantly.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import cohen_kappa_score

from sleepaccel.data.labels import CLASS_NAMES, INVALID
from sleepaccel.data.windows import evaluation_epoch_order

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CACHE = REPO_ROOT / "data" / "cache"

#: The level set in advance as "good enough". On Fleiss's widely used scale for
#: kappa, this is where agreement stops being rated poor.
TARGET = 0.40

NAME = {
    "accel_only": "Motion only",
    "accel_hr": "Motion + heart rate",
    "hr_only": "Heart rate only",
    "accel_only_nocontext": "Motion only, no night context",
}
MODEL_ORDER = ["accel_only_nocontext", "accel_only", "hr_only", "accel_hr"]
MAIN_THREE = ["accel_only", "hr_only", "accel_hr"]
MODEL_COLOR = {
    "accel_only": "#4c78a8",
    "hr_only": "#f58518",
    "accel_hr": "#54a24b",
    "accel_only_nocontext": "#a0a0a0",
}
STAGE_COLOR = {"Wake": "#e45756", "Light": "#8fb8de", "Deep": "#2c3e7a", "REM": "#b279a2"}
STAGE_PHRASE = {"Wake": "time awake", "Light": "Light sleep", "Deep": "Deep sleep", "REM": "REM sleep"}
STAGE_TEXT = {
    "Wake": "Awake in bed: before falling asleep, or brief wake-ups during the night. "
            "The body tends to move.",
    "Light": "Most of the night (clinically, stages N1 and N2). Easy to wake from, "
             "with occasional small movements.",
    "Deep": "Slow-wave sleep (N3). The hardest stage to wake from, and important for "
            "physical recovery. The body lies very still and the heart beats slowly "
            "and steadily.",
    "REM": "Dreaming sleep. The brain is highly active but the body is temporarily "
           "paralysed, so it also lies very still. Heart rate rises and turns irregular.",
}

#: Hypnograms conventionally draw deeper sleep lower: Wake at the top, then REM,
#: then Light, with Deep at the bottom.
DISPLAY_Y = {0: 3, 1: 1, 2: 0, 3: 2}
DISPLAY_TICKS = ([0, 1, 2, 3], ["Deep", "Light", "REM", "Wake"])

ABLATION_LABELS = {
    "accel_hr - accel_only": "Adding heart rate to motion",
    "hr_only - accel_only": "Heart rate alone vs motion alone",
    "accel_hr - hr_only": "Adding motion to heart rate",
    "accel_only_nocontext - accel_only": "Taking away the night context",
}

st.set_page_config(page_title="Sleep staging from a wrist band", layout="wide")


# --- loading ----------------------------------------------------------------


@st.cache_data
def load_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted(RESULTS.glob("*/summary.json")):
        if not path.parent.name.startswith("_"):  # _superseded_* are retired runs
            runs[path.parent.name] = json.loads(path.read_text())
    return runs


@st.cache_data
def load_manifest() -> dict | None:
    paths = sorted(CACHE.glob("*/manifest.json"))
    return json.loads(paths[0].read_text()) if paths else None


@st.cache_data
def load_ablation() -> dict | None:
    path = RESULTS / "ablation.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_predictions(run_id: str) -> dict | None:
    path = RESULTS / run_id / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as handle:
        data = {k: handle[k] for k in handle.files}
    data["subjects"] = data["subjects"].astype(str)
    return data


@st.cache_data
def load_subject(subject_id: str) -> dict | None:
    paths = sorted(CACHE.glob(f"*/{subject_id}.npz"))
    if not paths:
        return None
    with np.load(paths[0]) as handle:
        return {k: handle[k] for k in ("labels", "keep", "epoch_start_s")}


@st.cache_data
def night_timeline(run_id: str, subject_id: str, context_len: int) -> dict | None:
    """One person's night on a real time axis: the lab's answer and the model's guess.

    Saved predictions carry no timestamps, so their positions are rebuilt with
    evaluation_epoch_order, which also drops the few end-of-night chunks that
    evaluation scored twice. If the rebuilt order does not line up exactly with
    the lab labels, this returns None rather than draw a shifted chart.
    """
    preds, subject = load_predictions(run_id), load_subject(subject_id)
    if preds is None or subject is None:
        return None
    mask = preds["subjects"] == subject_id
    y_true, y_pred = preds["y_true"][mask], preds["y_pred"][mask]
    labels = subject["labels"]
    order = evaluation_epoch_order(subject["keep"], context_len)
    if order.size != y_true.size or not np.array_equal(labels[order], y_true):
        return None

    unique, first = np.unique(order, return_index=True)
    model = np.full(labels.size, np.nan)
    model[unique] = y_pred[first]
    lab = np.where(labels == INVALID, np.nan, labels).astype(float)
    return {"hours": subject["epoch_start_s"] / 3600.0, "lab": lab, "model": model}


@st.cache_data
def evaluation_repeats(run_ids: tuple[str, ...]) -> dict | None:
    """How many evaluated chunks were scored twice, and what that does to kappa."""
    runs = load_runs()
    shifts, repeated, total = [], 0, 0
    for run_id in run_ids:
        preds = load_predictions(run_id)
        if preds is None:
            continue
        context_len = runs[run_id]["args"]["context_len"]
        first_seen = np.zeros(preds["y_true"].size, dtype=bool)
        for subject_id in np.unique(preds["subjects"]):
            subject = load_subject(subject_id)
            if subject is None:
                return None
            idx = np.flatnonzero(preds["subjects"] == subject_id)
            order = evaluation_epoch_order(subject["keep"], context_len)
            if order.size != idx.size:
                return None
            _, first = np.unique(order, return_index=True)
            first_seen[idx[first]] = True
        yt, yp = preds["y_true"], preds["y_pred"]
        labels = list(range(len(CLASS_NAMES)))
        shifts.append(abs(
            cohen_kappa_score(yt[first_seen], yp[first_seen], labels=labels)
            - cohen_kappa_score(yt, yp, labels=labels)
        ))
        repeated, total = int((~first_seen).sum()), int(yt.size)
    if not shifts:
        return None
    return {"repeated": repeated, "total": total, "max_kappa_shift": max(shifts)}


# --- small helpers ------------------------------------------------------------


def name(run_id: str) -> str:
    return NAME.get(run_id, run_id)


def kappa_grade(kappa: float) -> str:
    if kappa < 0.40:
        return "poor"
    if kappa < 0.75:
        return "fair to good"
    return "excellent"


def how_to_read(text: str) -> None:
    st.caption(f"**How to read this:** {text}")


def takeaway(text: str) -> None:
    st.info(f"**What this tells us:** {text}", icon=":material/lightbulb:")


def bottom_line(text: str) -> None:
    with st.container(border=True):
        st.markdown("#### Bottom line")
        st.markdown(text)


def show(fig: go.Figure, key: str) -> None:
    """Render a chart. The explicit key is required, not decorative: two identical
    figures (the same model picked on both sides of a comparison) would otherwise
    get the same auto-generated element ID, and Streamlit refuses to draw the page."""
    st.plotly_chart(fig, key=key, width="stretch", config={"displayModeBar": False})


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
    )
    return fig


# --- charts -------------------------------------------------------------------


def kappa_scale_chart(runs: dict, run_ids: list[str]) -> go.Figure:
    fig = go.Figure()
    for x0, x1, colour, label in (
        (0.0, 0.40, "rgba(228,87,86,0.10)", "poor"),
        (0.40, 0.75, "rgba(242,183,5,0.14)", "fair to good"),
        (0.75, 1.0, "rgba(84,162,75,0.14)", "excellent"),
    ):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colour, line_width=0, layer="below",
                      annotation_text=label, annotation_position="top",
                      annotation_font_color="#666")
    fig.add_vline(x=TARGET, line_dash="dash", line_color="#555")
    kappas = [runs[r]["pooled"]["kappa"] for r in run_ids]
    fig.add_trace(go.Scatter(
        x=kappas, y=[name(r) for r in run_ids], mode="markers+text",
        marker=dict(size=18, color=[MODEL_COLOR.get(r, "#888") for r in run_ids]),
        text=[f"  {k:.2f}" for k in kappas], textposition="middle right",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[0, 1], title="Agreement with the sleep lab (kappa): 0 = guessing, 1 = perfect"),
        yaxis=dict(title=""),
    )
    return style(fig, 320)


def model_bars(run_ids, values, fmt, y_title, y_range, target=None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[name(r) for r in run_ids], y=values,
        marker_color=[MODEL_COLOR.get(r, "#888") for r in run_ids],
        text=[fmt.format(v) for v in values], textposition="outside", cliponaxis=False,
    ))
    if target is not None:
        fig.add_hline(y=target, line_dash="dash", line_color="#555",
                      annotation_text=f"target {target:.2f}", annotation_position="top left")
    fig.update_layout(yaxis=dict(range=y_range, title=y_title), showlegend=False)
    return style(fig, 380)


def confusion_chart(summary: dict, title: str) -> tuple[go.Figure, np.ndarray]:
    cm = np.array(summary["pooled"]["confusion"], dtype=float)
    normed = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(
        z=normed, x=list(CLASS_NAMES), y=list(CLASS_NAMES), colorscale="Blues",
        zmin=0, zmax=1, showscale=False,
        text=[[f"{v:.0%}" for v in row] for row in normed], texttemplate="%{text}",
        textfont=dict(size=15),
        hovertemplate="Sleep lab said %{y}, model guessed %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="What the model guessed"),
        yaxis=dict(title="What the sleep lab said", autorange="reversed"),
    )
    return style(fig, 400), normed


def stage_f1_chart(runs: dict, run_ids: list[str]) -> go.Figure:
    fig = go.Figure()
    for r in run_ids:
        f1 = runs[r]["pooled"]["per_class_f1"]
        fig.add_trace(go.Bar(
            name=name(r), x=list(CLASS_NAMES), y=[f1[c] for c in CLASS_NAMES],
            marker_color=MODEL_COLOR.get(r, "#888"),
            text=[f"{f1[c]:.2f}" for c in CLASS_NAMES], textposition="outside",
            cliponaxis=False,
        ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], title="F1 score: 0 = never, 1 = perfect"),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    return style(fig, 430)


def ablation_chart(ablation: dict) -> tuple[go.Figure, list[str]]:
    keys = [k for k in ABLATION_LABELS if k in ablation]
    fig = go.Figure()
    for key in keys:
        entry, label = ablation[key], ABLATION_LABELS[key]
        colour = "#4c78a8" if entry["excludes_zero"] else "#a0a0a0"
        fig.add_trace(go.Scatter(
            x=[entry["ci_low"], entry["ci_high"]], y=[label, label], mode="lines",
            line=dict(color=colour, width=7), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[entry["delta_mean"]], y=[label], mode="markers+text",
            marker=dict(size=15, color=colour), text=[f"{entry['delta_mean']:+.3f}"],
            textposition="top center", showlegend=False,
            hovertemplate=f"{label}: %{{x:+.3f}} (likely range "
                          f"{entry['ci_low']:+.3f} to {entry['ci_high']:+.3f})<extra></extra>",
        ))
    fig.add_vline(x=0, line_color="#333", line_width=1.5,
                  annotation_text="no difference", annotation_position="bottom")
    fig.update_layout(xaxis=dict(title="Change in agreement (kappa)"),
                      yaxis=dict(autorange="reversed", title=""))
    return style(fig, 360), keys


def hypnogram_chart(timeline: dict, run_id: str) -> go.Figure:
    hours, lab, model = timeline["hours"], timeline["lab"], timeline["model"]

    def to_display(values: np.ndarray) -> np.ndarray:
        out = np.full(values.shape, np.nan)
        for cls, y in DISPLAY_Y.items():
            out[values == cls] = y
        return out

    both = ~np.isnan(lab) & ~np.isnan(model)
    wrong = both & (lab != model)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.82, 0.18], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=hours, y=to_display(lab), name="Sleep lab (the answer)", line_shape="hv",
        connectgaps=False, line=dict(color="#222", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=hours, y=to_display(model), name=f"Model's guess ({name(run_id)})",
        line_shape="hv", connectgaps=False,
        line=dict(color=MODEL_COLOR.get(run_id, "#888"), width=2, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=hours[wrong], y=np.ones(int(wrong.sum())), width=30 / 3600,
        marker_color="#e45756", name="Mistake",
        hovertemplate="%{x:.2f} h<extra>mistake</extra>",
    ), row=2, col=1)
    fig.update_yaxes(tickvals=DISPLAY_TICKS[0], ticktext=DISPLAY_TICKS[1],
                     range=[-0.4, 3.4], row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="Mistakes", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Hours since the recording started", row=2, col=1)
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0), bargap=0)
    return style(fig, 480)


def training_chart(summary: dict, key: str, title: str, y_title: str, colour: str):
    histories = [f["history"] for f in summary["folds"] if f.get("history")]
    rounds = [h["epoch"] for h in histories[0]]
    values = np.array([[h[key] for h in history] for history in histories])
    fig = go.Figure()
    for i, row in enumerate(values, start=1):
        fig.add_trace(go.Scatter(
            x=rounds, y=row, mode="lines", line=dict(color=colour, width=1), opacity=0.35,
            showlegend=False,
            hovertemplate=f"test group {i}, round %{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=rounds, y=values.mean(axis=0), mode="lines+markers",
        line=dict(color=colour, width=3.5), showlegend=False,
        hovertemplate="average, round %{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=15)),
                      xaxis=dict(title="Training round", dtick=1), yaxis=dict(title=y_title))
    return style(fig, 340), rounds, values


# --- page ---------------------------------------------------------------------

runs = load_runs()
manifest = load_manifest()
ablation = load_ablation()
available = [r for r in MODEL_ORDER if r in runs] + sorted(set(runs) - set(MODEL_ORDER))
have_main = all(r in runs for r in MAIN_THREE)

st.title("Can a cheap fitness band tell which stage of sleep you're in?")
st.markdown(
    "An AI model trained on wrist-band data from 31 people, and checked against a "
    "hospital sleep test recorded the same night. **New here? Start with the first "
    "tab.** Every chart comes with a note on how to read it and what it shows, and "
    "every technical term is explained in the **Glossary** tab."
)

with st.sidebar:
    st.markdown("### In one sentence")
    st.write(
        "Motion alone can tell *whether* you slept, but not *how deeply*: "
        "for that you need heart rate."
    )
    st.markdown("### Three terms you'll see")
    st.markdown(
        "- **30-second chunk**: sleep is scored in 30-second pieces of the night.\n"
        "- **Kappa (κ)**: agreement with the sleep lab beyond lucky guessing. "
        "0 = guessing, 1 = perfect.\n"
        "- **F1**: how reliably one sleep stage is recognised, from 0 to 1."
    )
    st.caption(
        "Data: Walch et al. (2019), PhysioNet, open licence.  \n"
        "Code: github.com/ameyypawar/sleep-accel-dl"
    )

if not runs:
    st.warning(
        "No results yet. Train a model first: "
        "`python scripts/train.py --variant accel_only`"
    )
    st.stop()

tabs = st.tabs([
    "Start here",
    "The data",
    "Can motion alone do it?",
    "What does heart rate add?",
    "One night, up close",
    "How the model learned",
    "Glossary",
])


# --- 1. Start here -----------------------------------------------------------

with tabs[0]:
    st.header("The project in one minute")
    st.markdown(
        "**The question.** Many cheap fitness bands only have a motion sensor, with "
        "no heart-rate sensor, yet they still report how much *deep* and *REM* sleep "
        "you got. Can motion alone really tell those stages apart?\n\n"
        "**Why it matters.** Deep sleep and dreaming (REM) sleep do different jobs "
        "for the body and brain. If a band can't tell them apart, its breakdown of "
        "your night is guesswork.\n\n"
        "**What we did.** We trained an AI model on wrist recordings from 31 adults "
        "who also did a full hospital sleep test the same night. The hospital test "
        "is the answer key. We trained the model three ways (motion only, heart "
        "rate only, and both) to measure exactly what each signal contributes."
    )

    if have_main:
        k_motion = runs["accel_only"]["pooled"]["kappa"]
        k_both = runs["accel_hr"]["pooled"]["kappa"]
        b_motion = runs["accel_only"]["pooled"]["binary_kappa"]
        b_heart = runs["hr_only"]["pooled"]["binary_kappa"]

        st.subheader("The short answer")
        c1, c2, c3 = st.columns(3)
        with c1, st.container(border=True):
            st.markdown("**1. Motion alone isn't enough**")
            st.metric("Agreement with the sleep lab (κ)", f"{k_motion:.2f}",
                      help="0 means no better than guessing, 1 means a perfect match.")
            st.caption(f"Rated '{kappa_grade(k_motion)}'. The target was {TARGET:.2f}.")
        with c2, st.container(border=True):
            st.markdown("**2. Heart rate is the key ingredient**")
            st.metric("Agreement with heart rate added (κ)", f"{k_both:.2f}",
                      delta=f"{k_both - k_motion:+.2f} vs motion only",
                      help="The same model given both motion and heart rate.")
            st.caption("A big jump, although still short of the target.")
        with c3, st.container(border=True):
            st.markdown("**3. But motion is better at one job**")
            st.metric("Awake-or-asleep agreement, motion only (κ)", f"{b_motion:.2f}",
                      delta=f"{b_motion - b_heart:+.2f} vs heart rate only",
                      help="Only asking 'awake or asleep?' instead of all four stages.")
            st.caption("Motion spots when you're awake. Heart rate tells the sleep stages apart.")

        st.subheader("Where each version of the model lands")
        st.markdown(
            "**Kappa** scores how well the model's guesses match the sleep lab, after "
            "removing the matches you'd get by lucky guessing. **0 means no better than "
            "guessing; 1 means a perfect match.** The shaded bands are a widely used "
            "rule of thumb for reading it."
        )
        kappa_runs = [r for r in MODEL_ORDER if r in runs]
        show(kappa_scale_chart(runs, kappa_runs), key="kappa_scale")
        how_to_read(
            "Each dot is one version of the model; further right is better. The "
            f"dashed line is the target of {TARGET:.2f} that we set before running "
            "the experiment, the point where agreement stops being rated 'poor'."
        )
        best = max(kappa_runs, key=lambda r: runs[r]["pooled"]["kappa"])
        reached = [r for r in kappa_runs if runs[r]["pooled"]["kappa"] >= TARGET]
        takeaway(
            ("No version of the model reaches the target when asked to tell all four "
             "stages apart. " if not reached else
             f"{', '.join(name(r) for r in reached)} reached the target. ")
            + f"Adding heart rate moves the score a long way, from {k_motion:.2f} to "
            f"{k_both:.2f}, while motion alone barely gets off the floor. The best "
            f"version is **{name(best)}**."
        )

    st.subheader("First, what are the four sleep stages?")
    cols = st.columns(4)
    for col, stage in zip(cols, CLASS_NAMES):
        with col, st.container(border=True):
            st.markdown(
                f"<span style='color:{STAGE_COLOR[stage]};font-weight:700;"
                f"font-size:1.15rem'>{stage}</span>", unsafe_allow_html=True
            )
            st.write(STAGE_TEXT[stage])
    st.markdown(
        "**The key point for this project:** in both Deep sleep and REM sleep the "
        "body lies almost perfectly still, so a motion sensor sees nearly the same "
        "thing. The heart behaves very differently in the two: slow and steady in "
        "Deep sleep, faster and irregular in REM. That difference turns out to be "
        "the whole story."
    )

    st.subheader("How it works")
    steps = [
        ("The watch records", "How the wrist moves, in three directions, many times "
                              "a second. Plus heart rate."),
        ("Cut into 30-second chunks", "Sleep labs score the night in 30-second pieces, "
                                      "so the model works the same way."),
        ("The model reads each chunk", "Together with the 10 minutes around it, because "
                                       "sleep follows a pattern through the night."),
        ("It guesses the stage", "Wake, Light, Deep or REM, for every chunk of the night."),
        ("We check the answer", "Against the hospital sleep test, which experts "
                                "scored by hand."),
    ]
    layout = st.columns([4, 1, 4, 1, 4, 1, 4, 1, 4])
    for i, (title, text) in enumerate(steps):
        with layout[i * 2], st.container(border=True):
            st.markdown(f"**{i + 1}. {title}**")
            st.caption(text)
        if i < len(steps) - 1:
            layout[i * 2 + 1].markdown(
                "<div style='text-align:center;font-size:1.8rem;padding-top:2.4rem;"
                "color:#888'>&rarr;</div>", unsafe_allow_html=True
            )

    st.markdown(
        "**How we kept the test fair.** The model was always graded on people it had "
        "never seen during training. We split the 31 people into five groups, trained "
        "on four and tested on the fifth, and rotated until everyone had been tested "
        "once. Testing on people it had trained on would let it recognise individuals "
        "instead of learning about sleep, and would make the results look far better "
        "than they are."
    )

    with st.expander("For the technically curious: the model and how it was trained"):
        args = runs[available[0]]["args"]
        st.markdown(
            "- **Input:** each 30-second chunk of triaxial acceleration plus its vector "
            "magnitude, resampled to 30 Hz (4 channels × 900 samples).\n"
            "- **Encoder:** a 1D convolutional network (3 blocks of Conv1d k=7, "
            "BatchNorm, GELU, MaxPool, Dropout), then additive attention pooling over "
            "time to give one embedding per chunk.\n"
            f"- **Context:** a bidirectional LSTM over {args['context_len']} consecutive "
            f"chunks ({args['context_len'] * 30 // 60} minutes).\n"
            "- **Heart rate:** per-chunk mean, standard deviation, minimum, maximum and "
            "deviation from the person's median, plus a flag for chunks with no reading "
            "(so 'no reading' is never mistaken for 0 bpm).\n"
            f"- **Training:** AdamW, learning rate {args['lr']}, cosine schedule, "
            f"{args['epochs']} rounds, cross-entropy weighted by inverse class frequency "
            f"to the power {args.get('class_weight_power', 1.0)}. The saved version is "
            "the one with the best macro-F1 on held-back validation people.\n"
            "- **Evaluation:** subject-wise 5-fold cross-validation, with validation "
            "people drawn from the training side of each split; differences between "
            "versions tested with a paired bootstrap over people (1,000 resamples)."
        )
        repeats = evaluation_repeats(tuple(r for r in MODEL_ORDER if r in runs))
        if repeats and repeats["repeated"]:
            st.caption(
                f"Known quirk: the last window of each recording overlaps the one "
                f"before it, so {repeats['repeated']:,} of {repeats['total']:,} "
                f"evaluated chunks ({repeats['repeated'] / repeats['total']:.1%}) were "
                f"scored twice. Removing the repeats moves any kappa by at most "
                f"{repeats['max_kappa_shift']:.3f}, so no conclusion changes. The "
                "night-by-night chart in 'One night, up close' already removes them."
            )


# --- 2. The data --------------------------------------------------------------

with tabs[1]:
    st.header("The data")
    st.markdown(
        "The recordings come from a published study (Walch et al., 2019). **31 adults** "
        "wore an Apple Watch while doing a full hospital sleep test "
        "(**polysomnography**) on the same night. The watch recorded wrist motion and "
        "heart rate. Sleep experts scored the hospital recording by hand, one "
        "30-second chunk at a time, and that hand-scoring is the answer key the "
        "model is graded against."
    )

    if manifest is None:
        st.info("The data summary isn't built yet. Run `python -m sleepaccel.data.build_cache`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People", manifest["n_subjects"], help="Adults who took part in the study.")
        c2.metric("30-second chunks recorded", f"{manifest['n_epochs']:,}",
                  help="Every 30-second piece of every night, across all 31 people.")
        c3.metric("Usable chunks", f"{manifest['n_kept']:,}",
                  help="Chunks that passed the quality check and were scored by the lab.")
        c4.metric("Share kept", f"{manifest['kept_fraction']:.1%}")

        st.subheader("How a night divides up")
        st.markdown("Every usable chunk carries one of four labels, and they are far from evenly split.")
        counts = {k: v for k, v in manifest["class_counts"].items() if k != "Invalid"}
        total = sum(counts.values())
        fig = go.Figure(go.Bar(
            x=list(counts), y=list(counts.values()),
            marker_color=[STAGE_COLOR[k] for k in counts],
            text=[f"{v:,}<br>({v / total:.0%})" for v in counts.values()],
            textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(yaxis=dict(title="Number of 30-second chunks",
                                     range=[0, max(counts.values()) * 1.25]),
                          showlegend=False)
        show(style(fig, 380), key="class_distribution")
        how_to_read("Each bar counts the 30-second chunks with that label, across all 31 people.")
        share = {k: v / total for k, v in counts.items()}
        takeaway(
            f"A typical night is mostly Light sleep ({share['Light']:.0%} of chunks), "
            f"while Deep sleep is only {share['Deep']:.0%}. That has two consequences. "
            "First, plain **accuracy is misleading**: a model that answers 'Light' "
            f"every single time would be right {share['Light']:.0%} of the time while "
            "being useless, which is why we score with kappa instead. Second, the "
            "rarer stages give the model fewer examples to learn from."
        )

        st.subheader("Why some chunks were thrown out")
        st.markdown(
            "Before training, every chunk went through a **quality check**: did the "
            "watch actually record enough motion during those 30 seconds? Chunks with "
            "big gaps were left out rather than filled in. That matters: a gap filled "
            "in smoothly looks like a perfectly still body, which is exactly what deep "
            "sleep looks like. Filling the gaps would have taught the model that "
            "'missing data' means 'deep sleep'."
        )
        people = sorted(manifest["subjects"].values(),
                        key=lambda s: s["n_kept"] / max(s["n_epochs"], 1))
        fracs = [s["n_kept"] / max(s["n_epochs"], 1) for s in people]
        fig = go.Figure(go.Bar(
            x=[s["subject_id"] for s in people], y=[f * 100 for f in fracs],
            marker_color=["#e45756" if f < 0.8 else "#8fb8de" for f in fracs],
            hovertemplate="Person %{x}: %{y:.0f}% usable<extra></extra>",
        ))
        fig.update_xaxes(type="category", tickangle=-60, title="Person (study ID)")
        fig.update_yaxes(title="Share of the night usable (%)", range=[0, 105])
        show(style(fig, 380), key="people_usable")
        how_to_read(
            "One bar per person, sorted from least to most usable data. Red bars are "
            "people where more than a fifth of the night was unusable."
        )
        worst = people[0]
        no_signal = worst["n_signal_rejected"] / max(worst["n_epochs"], 1)
        takeaway(
            f"Most people kept around {np.median(fracs):.0%} of their night. The clear "
            f"outlier is person {worst['subject_id']}, whose watch recorded no usable "
            f"motion for {no_signal:.0%} of the night. The quality check caught it, "
            "instead of feeding the model an invented signal."
        )

        with st.expander("See the numbers for every person"):
            how_to_read(
                "One row per person, least usable data first. 'Watch readings per "
                "second' is how often the watch sampled motion; 'heart-rate coverage' "
                "is the share of the night with at least one heart-rate reading."
            )
            rates = [s["sample_rate_hz"] for s in people]
            st.dataframe(
                [
                    {
                        "Person": s["subject_id"],
                        "Chunks recorded": s["n_epochs"],
                        "Chunks kept": s["n_kept"],
                        "Share kept (%)": round(100 * s["n_kept"] / max(s["n_epochs"], 1), 1),
                        "Watch readings per second": round(s["sample_rate_hz"], 1),
                        "Heart-rate coverage (%)": round(100 * s["hr_valid_fraction"], 1),
                    }
                    for s in people
                ],
                width="stretch", hide_index=True,
            )
            st.caption(
                f"Watches took between {min(rates):.0f} and {max(rates):.0f} readings per "
                "second depending on the person, so every recording was converted to "
                "the same 30 readings per second before training."
            )


# --- 3. Can motion alone do it? ------------------------------------------------

with tabs[2]:
    st.header("Can motion alone tell the sleep stages apart?")
    st.markdown(
        "This is the main question. Here the model sees **only the wrist motion**, "
        f"with no heart rate. We set a target in advance: a kappa of **{TARGET:.2f}**, "
        "the point where agreement with the sleep lab stops being rated 'poor'."
    )

    if "accel_only" in runs:
        motion = runs["accel_only"]["pooled"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Agreement (κ)", f"{motion['kappa']:.2f}",
                  help=f"0 = no better than guessing, 1 = perfect. Target {TARGET:.2f}.")
        c2.metric("Rating", kappa_grade(motion["kappa"]).capitalize(),
                  help="Rule of thumb: below 0.40 poor, 0.40 to 0.75 fair to good, above 0.75 excellent.")
        c3.metric("Gap to perfect closed", f"{motion['kappa']:.0%}",
                  help="Kappa can be read as the share of the distance from pure "
                       "guessing to a perfect match that the model covers.")
        c4.metric("Awake-or-asleep only (κ)", f"{motion['binary_kappa']:.2f}",
                  help="The same score when only asking 'awake or asleep?' "
                       "rather than all four stages.")
        if motion["is_degenerate"]:
            st.warning(
                "This version of the model almost never predicts one of the stages, "
                "so its scores should be read with care."
            )

    st.subheader("Where the model gets confused")
    st.markdown(
        "A **confusion table** lines up the model's guesses against the sleep lab's "
        "answers. Pick two versions of the model to compare side by side."
    )
    left_col, right_col = st.columns(2)
    left_default = available.index("accel_only") if "accel_only" in available else 0
    right_default = available.index("accel_hr") if "accel_hr" in available else 0
    left_run = left_col.selectbox("Left", available, index=left_default,
                                  format_func=name, key="cm_left")
    right_run = right_col.selectbox("Right", available, index=right_default,
                                    format_func=name, key="cm_right")
    with left_col:
        fig, left_cm = confusion_chart(runs[left_run], name(left_run))
        show(fig, key="cm_left_chart")
    with right_col:
        fig, right_cm = confusion_chart(runs[right_run], name(right_run))
        show(fig, key="cm_right_chart")
    how_to_read(
        "Each row is what the sleep lab said; each column is what the model guessed. "
        "The percentages in a row add up to 100%. The diagonal from top left to bottom "
        "right is where the model was right, so the darker the diagonal, the better."
    )
    deep, light, rem = CLASS_NAMES.index("Deep"), CLASS_NAMES.index("Light"), CLASS_NAMES.index("REM")
    takeaway(
        f"Look at the **Deep** row. {name(left_run)} labels real deep sleep correctly "
        f"only {left_cm[deep, deep]:.0%} of the time, and mistakes it for Light sleep "
        f"{left_cm[deep, light]:.0%} of the time. {name(right_run)} gets it right "
        f"{right_cm[deep, deep]:.0%} of the time. The **REM** row tells the same story: "
        f"{left_cm[rem, rem]:.0%} correct against {right_cm[rem, rem]:.0%}."
    )

    st.subheader("How well each stage is recognised")
    st.markdown(
        "The **F1 score** rates how reliably the model recognises one stage. It rewards "
        "finding every real case of that stage *and* not raising false alarms, and "
        "runs from 0 (never gets it) to 1 (perfect)."
    )
    show(stage_f1_chart(runs, available), key="stage_f1")
    how_to_read(
        "Each group is one sleep stage; each coloured bar is one version of the model. "
        "Taller is better."
    )
    if have_main:
        mf1 = runs["accel_only"]["pooled"]["per_class_f1"]
        hf1 = runs["hr_only"]["pooled"]["per_class_f1"]
        takeaway(
            f"With motion alone, Deep sleep scores {mf1['Deep']:.2f} and REM "
            f"{mf1['REM']:.2f}: close to zero, meaning the model can barely find them. "
            f"Heart rate alone scores {hf1['Deep']:.2f} and {hf1['REM']:.2f}. For "
            f"**Wake** it's the other way round: motion scores {mf1['Wake']:.2f}, heart "
            f"rate only {hf1['Wake']:.2f}."
        )

    if "accel_only" in runs and "accel_only_nocontext" in runs:
        st.subheader("Does looking at the surrounding minutes help?")
        st.markdown(
            "The model normally reads each 30-second chunk together with the 10 minutes "
            "around it. To check that this matters, we also trained a version that sees "
            "each chunk entirely on its own."
        )
        pair = ["accel_only", "accel_only_nocontext"]
        st.dataframe(
            [
                {
                    "Version": name(r),
                    "Agreement (κ)": round(runs[r]["pooled"]["kappa"], 3),
                    "Deep sleep F1": round(runs[r]["pooled"]["per_class_f1"]["Deep"], 3),
                    "REM F1": round(runs[r]["pooled"]["per_class_f1"]["REM"], 3),
                }
                for r in pair
            ],
            width="stretch", hide_index=True,
        )
        how_to_read(
            "Each row is one version of the model, and higher is better in every "
            "column. The two F1 columns show how well it recognises Deep sleep and REM."
        )
        with_ctx, without = runs["accel_only"]["pooled"], runs["accel_only_nocontext"]["pooled"]
        takeaway(
            "Without the surrounding minutes, the model almost never picks Deep or REM "
            f"(F1 scores: Deep {without['per_class_f1']['Deep']:.2f}, REM "
            f"{without['per_class_f1']['REM']:.2f}); it falls back on the stages it can "
            "see directly. The context is what lets it attempt the hard stages at all, "
            f"even though it only lifts overall agreement from {without['kappa']:.2f} "
            f"to {with_ctx['kappa']:.2f}."
        )

    if "accel_only" in runs:
        k = runs["accel_only"]["pooled"]["kappa"]
        bottom_line(
            f"**No.** With motion alone the model reaches a kappa of {k:.2f}, rated "
            f"'{kappa_grade(k)}' and well short of the {TARGET:.2f} target. It can see "
            "when someone is moving, but Deep sleep and REM both look like 'lying "
            "perfectly still' to a motion sensor, so it cannot tell them apart."
        )


# --- 4. What does heart rate add? -----------------------------------------------

with tabs[3]:
    st.header("What does heart rate add?")
    st.markdown(
        "Same model, same people, same test. The only thing that changes is what the "
        "model is allowed to see: **motion only**, **heart rate only**, or **both**."
    )

    if not have_main:
        st.info("Train all three versions (motion only, heart rate only, both) to see this comparison.")
    else:
        st.subheader("Overall agreement for each version")
        kappas = [runs[r]["pooled"]["kappa"] for r in MAIN_THREE]
        show(model_bars(MAIN_THREE, kappas, "{:.2f}", "Agreement with the sleep lab (κ)",
                        [0, 0.5], target=TARGET), key="hr_kappa")
        how_to_read("Each bar is one version of the model; higher is better. The dashed line is the target.")
        km, kh, kb = kappas
        takeaway(
            f"Adding heart rate lifts agreement from {km:.2f} to {kb:.2f}, about "
            f"{kb / km:.1f} times higher. Heart rate on its own ({kh:.2f}) already does "
            "almost as well as both together. Even the best version stays below the target."
        )

        if ablation:
            st.subheader("Is the difference real, or just luck?")
            st.markdown(
                "With only 31 people, a difference could come down to which people "
                "happened to be in the test. To check, we re-ran each comparison "
                "**1,000 times**, each time on a random re-draw of the people, and "
                "recorded how big the difference came out. The line shows the range "
                "covering the middle 95% of those results. **If the whole line sits "
                "on one side of zero, the difference is real.** If it crosses zero, we "
                "can't rule out that there is no difference at all."
            )
            fig, keys = ablation_chart(ablation)
            show(fig, key="ablation")
            how_to_read(
                "The dot is the typical change in kappa; the line is its likely range. "
                "Blue lines are clearly real differences. Grey lines cross zero, so the "
                "difference could be down to luck."
            )
            sentences = []
            if "accel_hr - accel_only" in ablation:
                e = ablation["accel_hr - accel_only"]
                verdict = "a large, real improvement" if e["excludes_zero"] else "not a reliable difference"
                sentences.append(f"Adding heart rate to motion is {verdict} ({e['delta_mean']:+.2f}).")
            if "accel_hr - hr_only" in ablation:
                e = ablation["accel_hr - hr_only"]
                if e["excludes_zero"]:
                    sentences.append(
                        f"Adding motion to heart rate also helps a little but reliably "
                        f"({e['delta_mean']:+.3f})."
                    )
                else:
                    sentences.append(
                        f"But adding motion to heart rate gives {e['delta_mean']:+.3f}, "
                        f"with a likely range from {e['ci_low']:+.3f} to {e['ci_high']:+.3f}. "
                        "That crosses zero, so once heart rate is available, motion adds "
                        "nothing we can reliably detect."
                    )
            takeaway(" ".join(sentences))

        st.subheader("The twist: each signal is good at a different job")
        st.markdown(
            "The overall score hides something interesting. Split the problem in two "
            "(*is this person awake or asleep?* and *which stage of sleep are they "
            "in?*) and the two signals swap places."
        )
        binary = [runs[r]["pooled"]["binary_kappa"] for r in MAIN_THREE]
        deep_rem = [
            (runs[r]["pooled"]["per_class_f1"]["Deep"] + runs[r]["pooled"]["per_class_f1"]["REM"]) / 2
            for r in MAIN_THREE
        ]
        c1, c2 = st.columns(2)
        with c1:
            fig = model_bars(MAIN_THREE, binary, "{:.2f}", "Agreement (κ)", [0, 0.7])
            fig.update_layout(title=dict(text="Awake or asleep?", font=dict(size=15)))
            show(fig, key="twist_binary")
        with c2:
            fig = model_bars(MAIN_THREE, deep_rem, "{:.2f}", "Average F1 of Deep and REM", [0, 0.8])
            fig.update_layout(title=dict(text="Which stage of sleep? (Deep and REM)", font=dict(size=15)))
            show(fig, key="twist_stages")
        how_to_read(
            "Left: agreement on the simple awake-or-asleep question. Right: how well "
            "Deep sleep and REM are recognised (the average of their F1 scores). "
            "Higher is better on both."
        )
        takeaway(
            f"Motion is the better awake-or-asleep detector (kappa {binary[0]:.2f}, "
            f"against {binary[1]:.2f} for heart rate): a moving body is an awake body. "
            f"Heart rate is far better at telling sleep stages apart (Deep and REM F1 "
            f"{deep_rem[1]:.2f}, against {deep_rem[0]:.2f} for motion), because the "
            "heart behaves differently in Deep sleep and REM even when the body is "
            "still. Using both gives the best of each."
        )

        bottom_line(
            "Heart rate is what makes sleep-stage detection work. A band with only a "
            "motion sensor can estimate **how long** you slept, but its **deep sleep and "
            "REM** figures should be treated with scepticism."
        )


# --- 5. One night, up close ---------------------------------------------------------

with tabs[4]:
    st.header("One night, up close")
    st.markdown(
        "A **hypnogram** is a map of one night's sleep: time runs from left to right, "
        "and the line steps between stages as the night goes on. Sleep moves through "
        "cycles of roughly 90 minutes. Here the sleep lab's answer is drawn against "
        "the model's guess for one person, so you can see where it goes right and wrong."
    )

    runs_with_preds = [r for r in available if load_predictions(r) is not None]
    if not runs_with_preds:
        st.info("No saved predictions to show.")
    else:
        c1, c2 = st.columns(2)
        run_id = c1.selectbox(
            "Version of the model", runs_with_preds,
            index=runs_with_preds.index("accel_hr") if "accel_hr" in runs_with_preds else 0,
            format_func=name, key="night_run",
        )
        preds = load_predictions(run_id)
        people = sorted(np.unique(preds["subjects"]).tolist(), key=int)
        busiest = max(people, key=lambda p: int((preds["subjects"] == p).sum()))
        subject_id = c2.selectbox(
            "Person", people, index=people.index(busiest),
            format_func=lambda p: f"Person {p}", key="night_person",
        )

        timeline = night_timeline(run_id, subject_id, runs[run_id]["args"]["context_len"])
        if timeline is None:
            st.warning(
                "Couldn't line this person's predictions up against the night's timeline, "
                "so the chart is hidden rather than drawn in the wrong place."
            )
        else:
            lab, model, hours = timeline["lab"], timeline["model"], timeline["hours"]
            both = ~np.isnan(lab) & ~np.isnan(model)
            agree = float((lab[both] == model[both]).mean()) if both.any() else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Length of recording", f"{hours[-1] + 30 / 3600 - hours[0]:.1f} hours")
            m2.metric("Chunks the model scored", f"{int(both.sum()):,}")
            m3.metric("Agreement with the sleep lab", f"{agree:.0%}",
                      help="Share of this person's scored chunks where the model's "
                           "guess matched the lab.")

            show(hypnogram_chart(timeline, run_id), key="hypnogram")
            how_to_read(
                "The solid black line is the sleep lab's answer; the dotted line is the "
                "model's guess. Deeper sleep is drawn lower. Where the dotted line has a "
                "gap, the watch had no usable data there and the model made no guess. "
                "The red marks underneath show every 30-second chunk the model got wrong."
            )

            wrong = np.flatnonzero(both & (lab != model))
            sentences = [f"The model agrees with the sleep lab on **{agree:.0%}** of this person's night."]
            if wrong.size:
                (true_cls, guess_cls), count = Counter(
                    (int(lab[i]), int(model[i])) for i in wrong
                ).most_common(1)[0]
                sentences.append(
                    f"Its most common mistake is calling "
                    f"{STAGE_PHRASE[CLASS_NAMES[true_cls]]} '{CLASS_NAMES[guess_cls]}' "
                    f"({count} chunks, {count / wrong.size:.0%} of its mistakes)."
                )
                labelled = ~np.isnan(lab)
                change = np.flatnonzero(labelled[1:] & labelled[:-1] & (lab[1:] != lab[:-1])) + 1
                near = np.zeros(lab.size, dtype=bool)
                for c in change:
                    near[max(0, c - 2):c + 2] = True
                near_share, base = float(near[wrong].mean()), float(near[both].mean())
                if near_share > 1.5 * base:
                    sentences.append(
                        f"{near_share:.0%} of its mistakes fall within a minute of a real "
                        f"change of stage, although only {base:.0%} of the night lies that "
                        "close to one, so its errors bunch up where one stage turns into the next."
                    )
                else:
                    sentences.append(
                        f"Only {near_share:.0%} of its mistakes fall within a minute of a "
                        f"real change of stage, roughly what chance would give ({base:.0%} "
                        "of the night lies that close to one). So the errors aren't just "
                        "about timing: the model mislabels whole stretches of the night."
                    )
            takeaway(" ".join(sentences))

            if "accel_only" in runs and manifest is not None:
                predicted_light = runs["accel_only"]["pooled"]["predicted_support"]["Light"]
                predicted_total = sum(runs["accel_only"]["pooled"]["predicted_support"].values())
                true_light = manifest["class_counts"]["Light"] / sum(
                    v for k, v in manifest["class_counts"].items() if k != "Invalid"
                )
                st.caption(
                    "Try switching the model to **Motion only** for the same person. Across "
                    f"everyone it answers 'Light' for {predicted_light / predicted_total:.0%} "
                    f"of chunks, when the true figure is {true_light:.0%}: with Deep and REM "
                    "looking the same as Light to a motion sensor, 'Light' is its safest bet."
                )


# --- 6. How the model learned -------------------------------------------------

with tabs[5]:
    st.header("How the model learned")
    st.markdown(
        "The model learns in **training rounds**. In each round it works through all "
        "the training data once and adjusts itself to make fewer mistakes. After every "
        "round it is checked on a few people held back for that purpose, and we keep "
        "the version that scored best on that check rather than the last one. (In "
        "machine learning a training round is called an 'epoch'. That is not the same "
        "thing as the 30-second chunks, which sleep science also calls epochs.)"
    )
    run_id = st.selectbox(
        "Version of the model", available,
        index=available.index("accel_only") if "accel_only" in available else 0,
        format_func=name, key="learn_run",
    )
    summary = runs[run_id]
    if not any(f.get("history") for f in summary["folds"]):
        st.info("No training history was recorded for this version.")
    else:
        colour = MODEL_COLOR.get(run_id, "#4c78a8")
        c1, c2 = st.columns(2)
        with c1:
            fig, rounds, loss = training_chart(
                summary, "train_loss", "Mistakes on the training data",
                "Training error (lower is better)", colour)
            show(fig, key="train_loss")
        with c2:
            fig, _, val = training_chart(
                summary, "val_macro_f1", "Score on held-back people",
                "Macro F1 (higher is better)", colour)
            show(fig, key="train_val")
        how_to_read(
            "Each thin line is one of the five test groups; the thick line is their "
            "average. Left: lower means fewer mistakes on the data the model learns "
            "from. Right: higher means better results on people it is not learning from."
        )
        mean_loss, mean_val = loss.mean(axis=0), val.mean(axis=0)
        best_round = rounds[int(np.argmax(mean_val))]
        if len(rounds) >= 6:
            late_gain = float(mean_val[-1] - mean_val[4])
            span = f"rounds 6 to {rounds[-1]}"
            if late_gain >= 0.03:
                ending = (
                    f"and the score on held-back people was still clearly rising at the "
                    f"end ({late_gain:+.3f} over {span}), so more training might "
                    "improve this version."
                )
            elif best_round == rounds[-1] and late_gain > 0:
                ending = (
                    f"while the score on held-back people is still creeping up at the "
                    f"end, but slowly: {span} added only {late_gain:+.3f}. More training "
                    "might add a little, but the curve is flattening out."
                )
            else:
                ending = (
                    f"but the score on held-back people levels off: it peaked at round "
                    f"{best_round}, and {span} changed it by only {late_gain:+.3f}. The "
                    "model is getting better at the people it trains on without getting "
                    "better at new people, so training longer is unlikely to help."
                )
        else:
            ending = f"and the score on held-back people peaks at round {best_round}."
        takeaway(
            f"The training error keeps falling (from {mean_loss[0]:.2f} to "
            f"{mean_loss[-1]:.2f}), {ending}"
        )


# --- 7. Glossary ------------------------------------------------------------------

with tabs[6]:
    st.header("Glossary")
    st.markdown("Every technical term used in this dashboard, in plain English.")

    st.subheader("The sleep science")
    st.markdown(
        "- **Polysomnography (PSG):** the hospital sleep test. Sensors on the head, "
        "face and chest record brain waves, eye movements, muscle activity and "
        "breathing overnight, and experts then score each 30-second chunk by hand. It "
        "is the gold standard, and the answer key for this project.\n"
        "- **Sleep stages:** *Wake*; *Light* (clinical stages N1 and N2); *Deep* "
        "(stage N3, also called slow-wave sleep); and *REM* (rapid eye movement, or "
        "dreaming sleep). See the 'Start here' tab for what each one is like.\n"
        "- **30-second chunk (epoch):** sleep is scored in 30-second pieces of the "
        "night. This dataset has 27,211 of them across 31 people.\n"
        "- **Sleep cycle:** through the night, sleep repeats a pattern of stages "
        "roughly every 90 minutes.\n"
        "- **Hypnogram:** a chart of sleep stages across one night, with time "
        "running from left to right."
    )

    st.subheader("The measurements")
    st.markdown(
        "- **Accelerometer:** the motion sensor in a watch or band. It measures "
        "movement in three directions (up-down, left-right, forward-back), many "
        "times a second.\n"
        "- **Heart-rate sensor (PPG):** the green light on the back of many watches, "
        "which reads your pulse from blood flow under the skin. Cheaper bands often "
        "don't have one.\n"
        "- **Sampling rate:** how many readings a sensor takes per second. Here it "
        "varied from about 10 to 67 per second between watches, so every recording "
        "was converted to a common 30.\n"
        "- **Quality check:** before training, chunks where the watch recorded too "
        "little data were left out rather than filled in, because smoothly filled "
        "gaps look exactly like deep sleep."
    )

    st.subheader("How the model is scored")
    st.markdown(
        "- **Cohen's kappa (κ):** agreement between the model and the sleep lab, after "
        "subtracting the agreement you'd get by chance. 0 means no better than "
        "guessing; 1 means perfect. It can be read as the share of the gap between "
        "guessing and perfection that the model closes. The rule of thumb used here "
        "(Fleiss): below 0.40 is poor, 0.40 to 0.75 is fair to good, and above 0.75 "
        "is excellent.\n"
        "- **Awake-or-asleep kappa:** the same score when only asking 'awake or "
        "asleep?' rather than all four stages. An easier question.\n"
        "- **Accuracy:** the share of chunks labelled correctly. Misleading here: "
        "Light sleep makes up more than half of the night, so answering 'Light' "
        "every time would be right more than half the time while being useless.\n"
        "- **F1 score:** how reliably one stage is recognised, from 0 to 1. It "
        "balances catching every real case of that stage against raising false alarms.\n"
        "- **Macro F1:** the average F1 over the four stages, each counted equally, "
        "so the rare stages matter as much as the common ones.\n"
        "- **Confusion table:** a grid comparing the lab's answers (rows) with the "
        "model's guesses (columns). Correct answers sit on the diagonal.\n"
        "- **Cross-validation, by person:** testing only on people the model never "
        "trained on, rotating through five groups so everyone is tested once.\n"
        "- **Likely range (95% confidence interval):** the range the true difference "
        "very probably falls in. Found here by 'bootstrapping': re-drawing the 31 "
        "people at random 1,000 times and re-measuring each time. If the range "
        "doesn't cross zero, the difference is real."
    )

    st.subheader("How the model works")
    st.markdown(
        "- **Neural network:** a model that learns patterns from examples, rather "
        "than following rules written by hand.\n"
        "- **1D convolutional network (CNN):** the part that scans each 30-second "
        "chunk of motion for short, telling patterns: a twitch, a turn, a restless "
        "stretch.\n"
        "- **Attention:** lets the model focus on the few moments in a chunk that "
        "matter, instead of averaging over 30 seconds of mostly nothing.\n"
        "- **BiLSTM (night context):** the part that reads the run of chunks forwards "
        "and backwards across 10 minutes, so each guess takes account of what came "
        "before and after.\n"
        "- **Training round (epoch):** one full pass through the training data. "
        "Machine learning also calls this an 'epoch', which is not the same as a "
        "30-second chunk.\n"
        "- **Training error (loss):** the model's measure of its own mistakes while "
        "it learns. Lower is better.\n"
        "- **Class weighting:** giving the rarer stages extra weight during training, "
        "so the model doesn't just learn to answer 'Light'."
    )
