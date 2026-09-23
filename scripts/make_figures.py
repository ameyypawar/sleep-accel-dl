"""Render the report figures from saved results.

Reads only `results/` and the cache manifest, so figures always match the run
that produced them rather than a separately-recomputed version.

    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sleepaccel.data.labels import CLASS_NAMES
from sleepaccel.paths import REPO_ROOT

RESULTS = REPO_ROOT / "results"
FIGS = RESULTS / "figures"
CACHE = REPO_ROOT / "data" / "cache"

PALETTE = {"Wake": "#c44e52", "Light": "#4c72b0", "Deep": "#55a868", "REM": "#8172b2"}
VARIANT_LABEL = {
    "accel_only": "Accel only",
    "accel_only_nocontext": "Accel only\n(no context)",
    "hr_only": "HR only",
    "accel_hr": "Accel + HR",
}
ORDER = ["accel_only_nocontext", "accel_only", "hr_only", "accel_hr"]

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def load(run_id: str) -> dict | None:
    path = RESULTS / run_id / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def fig_class_distribution() -> None:
    manifest = json.loads(sorted(CACHE.glob("*/manifest.json"))[0].read_text())
    counts = {k: v for k, v in manifest["class_counts"].items() if k != "Invalid"}
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    bars = ax.bar(list(counts), list(counts.values()),
                  color=[PALETTE[k] for k in counts])
    for bar, value in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.012,
                f"{value:,}\n({value/total:.0%})", ha="center", fontsize=7.5)
    ax.set_ylabel("epochs")
    ax.set_ylim(0, max(counts.values()) * 1.22)
    ax.set_title("Class distribution over 25,285 usable epochs", fontsize=9.5)
    fig.savefig(FIGS / "class_distribution.png")
    plt.close(fig)


def fig_confusion() -> None:
    runs = [("accel_only", "Accelerometer only"), ("accel_hr", "Accelerometer + heart rate")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    for ax, (run_id, title) in zip(axes, runs):
        summary = load(run_id)
        cm = np.array(summary["pooled"]["confusion"], dtype=float)
        normed = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{normed[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if normed[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASS_NAMES, fontsize=8)
        ax.set_yticks(range(4), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}\n$\\kappa$ = {summary['pooled']['kappa']:.3f}", fontsize=9)

    fig.colorbar(im, ax=axes, shrink=0.8, label="row-normalised share")
    fig.savefig(FIGS / "confusion.png")
    plt.close(fig)


def fig_per_class_f1() -> None:
    summaries = {r: load(r) for r in ORDER if load(r)}
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(summaries)

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (run_id, summary) in enumerate(summaries.items()):
        f1 = summary["pooled"]["per_class_f1"]
        ax.bar(x + i * width - 0.4 + width / 2,
               [f1[c] for c in CLASS_NAMES], width,
               label=VARIANT_LABEL[run_id].replace("\n", " "))
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7.5, ncols=2, frameon=False)
    ax.set_title("Per-class F1: motion detects wake, heart rate identifies stage",
                 fontsize=9.5)
    fig.savefig(FIGS / "per_class_f1.png")
    plt.close(fig)


def fig_kappa_comparison() -> None:
    summaries = {r: load(r) for r in ORDER if load(r)}
    fig, ax = plt.subplots(figsize=(5.0, 2.8))

    names = [VARIANT_LABEL[r] for r in summaries]
    means = [s["across_folds"]["kappa_mean"] for s in summaries.values()]
    stds = [s["across_folds"]["kappa_std"] for s in summaries.values()]

    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color=["#bbbbbb", "#4c72b0", "#dd8452", "#55a868"])
    ax.axhline(0.40, ls="--", lw=1, color="#c44e52")
    ax.text(len(names) - 0.4, 0.412, "clinical threshold 0.40",
            fontsize=7.5, color="#c44e52", ha="right")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_ylim(0, 0.50)
    ax.set_title("Four-class agreement by input variant", fontsize=9.5)
    fig.savefig(FIGS / "kappa_comparison.png")
    plt.close(fig)


def fig_ablation() -> None:
    data = json.loads((RESULTS / "ablation.json").read_text())
    labels = {
        "accel_hr - accel_only": "HR added to motion",
        "hr_only - accel_only": "HR alone vs motion alone",
        "accel_hr - hr_only": "Motion added to HR",
        "accel_only_nocontext - accel_only": "Removing sequence context",
    }
    keys = [k for k in labels if k in data]

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    y = np.arange(len(keys))
    for i, key in enumerate(keys):
        entry = data[key]
        crosses_zero = not entry["excludes_zero"]
        color = "#999999" if crosses_zero else "#4c72b0"
        ax.plot([entry["ci_low"], entry["ci_high"]], [i, i], lw=2.5, color=color)
        ax.plot(entry["delta_mean"], i, "o", ms=6, color=color)
        ax.text(entry["ci_high"] + 0.012, i,
                f"{entry['delta_mean']:+.3f}" + ("  (n.s.)" if crosses_zero else ""),
                va="center", fontsize=7.5, color=color)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [labels[k] for k in keys], fontsize=8)
    ax.set_xlabel("$\\Delta\\kappa$ with 95% CI (paired bootstrap over subjects)")
    ax.set_xlim(-0.09, 0.38)
    ax.invert_yaxis()
    ax.set_title("Ablation: what each component contributes", fontsize=9.5)
    fig.savefig(FIGS / "ablation.png")
    plt.close(fig)


def fig_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for run_id, style in (("accel_only", "-"), ("accel_hr", "--")):
        summary = load(run_id)
        if not summary:
            continue
        epochs = [h["epoch"] for h in summary["folds"][0]["history"]]
        loss = np.mean([[h["train_loss"] for h in f["history"]] for f in summary["folds"]], axis=0)
        f1 = np.mean([[h["val_macro_f1"] for h in f["history"]] for f in summary["folds"]], axis=0)
        axes[0].plot(epochs, loss, style, label=VARIANT_LABEL[run_id])
        axes[1].plot(epochs, f1, style, label=VARIANT_LABEL[run_id])

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro F1")
    for ax in axes:
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("Mean across five folds", fontsize=9.5)
    fig.savefig(FIGS / "training_curves.png")
    plt.close(fig)


def fig_hypnogram() -> None:
    with np.load(RESULTS / "accel_hr" / "predictions.npz", allow_pickle=True) as handle:
        preds = {k: handle[k] for k in handle.files}

    # Pick the subject with the most epochs, for a full night.
    subjects, counts = np.unique(preds["subjects"], return_counts=True)
    subject = subjects[int(np.argmax(counts))]
    mask = preds["subjects"] == subject
    y_true, y_pred = preds["y_true"][mask], preds["y_pred"][mask]
    hours = np.arange(y_true.size) * 30.0 / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].step(hours, y_true, where="post", lw=1.2, label="PSG (ground truth)")
    axes[0].step(hours, y_pred, where="post", lw=1.0, ls="--", label="Predicted")
    axes[0].set_yticks(range(4), CLASS_NAMES, fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, ncols=2)
    axes[0].set_title(f"Subject {subject}, accelerometer + heart rate "
                      f"({(y_true == y_pred).mean():.0%} epoch agreement)", fontsize=9.5)

    axes[1].fill_between(hours, (y_true != y_pred).astype(int), step="post",
                         color="#c44e52", alpha=0.7)
    axes[1].set_yticks([0, 1], ["agree", "differ"], fontsize=7.5)
    axes[1].set_xlabel("hours from recording start")
    fig.savefig(FIGS / "hypnogram.png")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("class_distribution", fig_class_distribution),
        ("confusion", fig_confusion),
        ("per_class_f1", fig_per_class_f1),
        ("kappa_comparison", fig_kappa_comparison),
        ("ablation", fig_ablation),
        ("training_curves", fig_training_curves),
        ("hypnogram", fig_hypnogram),
    ]:
        fn()
        print(f"  wrote {FIGS / (name + '.png')}")


if __name__ == "__main__":
    main()
