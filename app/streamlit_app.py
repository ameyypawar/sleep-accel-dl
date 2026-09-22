"""Dashboard for the accelerometer-only sleep staging experiments.

    streamlit run app/streamlit_app.py

Reads only what the training runs wrote to `results/` and the cache manifest.
It never trains, and it imports torch lazily, so it opens instantly and cannot
be affected by a training job running alongside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sleepaccel.data.labels import CLASS_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CACHE = REPO_ROOT / "data" / "cache"

CLASS_COLORS = {"Wake": "#e45756", "Light": "#4c78a8", "Deep": "#54a24b", "REM": "#b279a2"}

st.set_page_config(page_title="Accelerometer-only sleep staging", layout="wide")


# --- loaders ---------------------------------------------------------------


@st.cache_data
def load_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted(RESULTS.glob("*/summary.json")):
        runs[path.parent.name] = json.loads(path.read_text())
    return runs


@st.cache_data
def load_manifest() -> dict | None:
    paths = sorted(CACHE.glob("*/manifest.json"))
    return json.loads(paths[0].read_text()) if paths else None


@st.cache_data
def load_predictions(run_id: str) -> dict | None:
    path = RESULTS / run_id / "predictions.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


@st.cache_data
def load_subject_epochs(subject_id: str) -> dict | None:
    paths = sorted(CACHE.glob(f"*/{subject_id}.npz"))
    if not paths:
        return None
    with np.load(paths[0]) as handle:
        return {k: handle[k] for k in ("labels", "keep", "epoch_start_s")}


runs = load_runs()
manifest = load_manifest()

st.title("Sleep staging from wrist accelerometer alone")
st.caption(
    "Four-class staging (Wake / Light / Deep / REM) with no heart rate and no EEG. "
    "Walch et al. 2019, 31 subjects, raw triaxial acceleration with concurrent "
    "polysomnography."
)

if not runs:
    st.warning("No runs found. Train one first: `python scripts/train.py --variant accel_only`")
    st.stop()

tabs = st.tabs(
    ["Dataset", "RQ1 results", "RQ2 heart-rate ablation", "Training curves", "Hypnogram"]
)


# --- dataset ---------------------------------------------------------------

with tabs[0]:
    st.header("Dataset")
    if manifest is None:
        st.info("No cache manifest yet. Run `python -m sleepaccel.data.build_cache`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subjects", manifest["n_subjects"])
        c2.metric("Epochs", f"{manifest['n_epochs']:,}")
        c3.metric("Usable", f"{manifest['n_kept']:,}")
        c4.metric("Retained", f"{manifest['kept_fraction']:.1%}")

        counts = {k: v for k, v in manifest["class_counts"].items() if k != "Invalid"}
        st.subheader("Class distribution")
        st.caption(
            "Light dominates, Deep is scarcest. A flat distribution here would "
            "indicate the labels are misaligned against the signal."
        )
        fig = px.bar(
            x=list(counts), y=list(counts.values()),
            color=list(counts), color_discrete_map=CLASS_COLORS,
            labels={"x": "", "y": "epochs"},
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-subject signal quality")
        st.caption(
            "Sample rate varies from 10 to 67 Hz across subjects, so everything "
            "is resampled onto a fixed grid. Subject 7749105 has no accelerometer "
            "data in 86% of its epochs; those are rejected rather than "
            "interpolated, because interpolating across a dropout produces a flat "
            "line that looks exactly like deep sleep."
        )
        rows = [
            {
                "subject": s["subject_id"],
                "epochs": s["n_epochs"],
                "kept": s["n_kept"],
                "retained": s["n_kept"] / max(s["n_epochs"], 1),
                "sample_rate_hz": s["sample_rate_hz"],
                "hr_coverage": s["hr_valid_fraction"],
            }
            for s in manifest["subjects"].values()
        ]
        rows.sort(key=lambda r: r["retained"])
        st.dataframe(rows, use_container_width=True, hide_index=True)


# --- RQ1 -------------------------------------------------------------------

with tabs[1]:
    st.header("RQ1 — can motion alone reach clinically useful agreement?")
    run_id = st.selectbox("Run", sorted(runs), key="rq1_run")
    summary = runs[run_id]
    pooled = summary["pooled"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohen's kappa", f"{pooled['kappa']:.3f}", help="Clinical 'fair agreement' starts at 0.40")
    c2.metric("Macro F1", f"{pooled['macro_f1']:.3f}")
    c3.metric("Accuracy", f"{pooled['accuracy']:.3f}")
    c4.metric("Binary kappa", f"{pooled['binary_kappa']:.3f}", help="Sleep vs wake")

    if pooled["is_degenerate"]:
        st.error(
            "Degenerate predictions: at least one class is never predicted, or one "
            "class swamps the rest. The metrics below are not meaningful."
        )

    threshold = 0.40
    if pooled["kappa"] >= threshold:
        st.success(f"kappa {pooled['kappa']:.3f} meets the 0.40 clinical threshold.")
    else:
        st.info(
            f"kappa {pooled['kappa']:.3f} is below the 0.40 clinical threshold. "
            "That is a result, not a failure -- it quantifies the ceiling of "
            "motion-only staging."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = np.array(pooled["confusion"], dtype=float)
        normed = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig = px.imshow(
            normed, x=list(CLASS_NAMES), y=list(CLASS_NAMES),
            labels={"x": "predicted", "y": "true", "color": "row share"},
            color_continuous_scale="Blues", text_auto=".2f", zmin=0, zmax=1,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Per-class F1")
        f1 = pooled["per_class_f1"]
        fig = px.bar(
            x=list(f1), y=list(f1.values()), color=list(f1),
            color_discrete_map=CLASS_COLORS, labels={"x": "", "y": "F1"},
        )
        fig.update_layout(showlegend=False, height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Across folds")
    across = summary["across_folds"]
    st.write(
        f"kappa {across['kappa_mean']:.3f} ± {across['kappa_std']:.3f} "
        f"over {across['n_folds']} subject-wise folds"
    )
    st.dataframe(
        [
            {
                "fold": f["fold"],
                "kappa": round(f["kappa"], 4),
                "macro_f1": round(f["macro_f1"], 4),
                "accuracy": round(f["accuracy"], 4),
                "degenerate": f["is_degenerate"],
                "test_subjects": ", ".join(map(str, f["test_subjects"])),
            }
            for f in summary["folds"]
        ],
        use_container_width=True, hide_index=True,
    )


# --- RQ2 -------------------------------------------------------------------

with tabs[2]:
    st.header("RQ2 — what does heart rate actually add?")
    st.caption(
        "Same architecture, same folds, three input variants. This is the "
        "contribution: not that motion works, but by how much heart rate "
        "improves on it."
    )

    arms = {k: v for k, v in runs.items() if v.get("variant") in ("accel_only", "accel_hr", "hr_only")}
    if len(arms) < 2:
        st.info("Train at least two variants to compare.")
    else:
        rows = [
            {
                "run": k,
                "variant": v["variant"],
                "kappa": round(v["pooled"]["kappa"], 4),
                "macro_f1": round(v["pooled"]["macro_f1"], 4),
                "accuracy": round(v["pooled"]["accuracy"], 4),
                "degenerate": v["pooled"]["is_degenerate"],
            }
            for k, v in sorted(arms.items())
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        fig = px.bar(
            x=[r["variant"] for r in rows], y=[r["kappa"] for r in rows],
            labels={"x": "", "y": "Cohen's kappa"}, text=[f"{r['kappa']:.3f}" for r in rows],
        )
        fig.add_hline(y=0.40, line_dash="dash",
                      annotation_text="clinical threshold 0.40")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        ablation = RESULTS / "ablation.json"
        if ablation.exists():
            data = json.loads(ablation.read_text())
            st.subheader("Paired bootstrap over subjects")
            st.caption(
                "Subjects are resampled, not epochs, because epochs within a "
                "subject are not independent. The same resample is applied to "
                "both arms, which is what makes the interval paired."
            )
            for name, entry in data.items():
                sig = "excludes zero" if entry["excludes_zero"] else "includes zero"
                st.write(
                    f"**{name}**: Δkappa = {entry['delta_mean']:+.4f} "
                    f"(95% CI {entry['ci_low']:+.4f} to {entry['ci_high']:+.4f}, {sig})"
                )
        else:
            st.info("Run `python scripts/ablation.py` for bootstrap confidence intervals.")


# --- training curves --------------------------------------------------------

with tabs[3]:
    st.header("Training curves")
    run_id = st.selectbox("Run", sorted(runs), key="curve_run")
    summary = runs[run_id]
    records = [
        {"fold": f["fold"], **h} for f in summary["folds"] for h in f.get("history", [])
    ]
    if not records:
        st.info("No history recorded for this run.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(records, x="epoch", y="train_loss", color="fold", markers=True)
            fig.update_layout(height=380, title="Training loss")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(records, x="epoch", y="val_macro_f1", color="fold", markers=True)
            fig.update_layout(height=380, title="Validation macro F1 (selection metric)")
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Checkpoints are selected on macro F1 rather than accuracy, because "
            "Light is 55% of epochs and accuracy rewards predicting it always."
        )


# --- hypnogram --------------------------------------------------------------

with tabs[4]:
    st.header("Hypnogram — predicted against polysomnography")
    run_id = st.selectbox("Run", sorted(runs), key="hyp_run")
    preds = load_predictions(run_id)

    if preds is None:
        st.info("No predictions saved for this run.")
    else:
        subjects = sorted(set(preds["subjects"].tolist()))
        subject = st.selectbox("Subject", subjects)
        mask = preds["subjects"] == subject
        y_true = preds["y_true"][mask]
        y_pred = preds["y_pred"][mask]

        agree = float((y_true == y_pred).mean()) if y_true.size else 0.0
        c1, c2 = st.columns(2)
        c1.metric("Epochs", f"{y_true.size:,}")
        c2.metric("Epoch agreement", f"{agree:.1%}")

        hours = np.arange(y_true.size) * 30.0 / 3600.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=y_true, name="PSG (ground truth)",
                                 line_shape="hv", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=hours, y=y_pred, name="Predicted",
                                 line_shape="hv", line=dict(width=2, dash="dot")))
        fig.update_layout(
            height=420, xaxis_title="hours from recording start",
            yaxis=dict(tickmode="array", tickvals=list(range(4)), ticktext=list(CLASS_NAMES)),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Where it disagrees")
        disagree = (y_true != y_pred).astype(int)
        fig = go.Figure(go.Scatter(x=hours, y=disagree, line_shape="hv", fill="tozeroy",
                                   line=dict(width=1)))
        fig.update_layout(height=180, yaxis=dict(tickvals=[0, 1], ticktext=["agree", "differ"]),
                          xaxis_title="hours")
        st.plotly_chart(fig, use_container_width=True)
