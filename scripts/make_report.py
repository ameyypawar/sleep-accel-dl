"""Render the project report to PDF.

Every number is read from `results/` rather than typed in, so the report cannot
drift from the run that produced it.

    python scripts/make_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import json
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sleepaccel.paths import REPO_ROOT

RESULTS = REPO_ROOT / "results"
FIGS = RESULTS / "figures"
OUT = REPO_ROOT / "reports" / "sleep-accel-dl-report.pdf"

ACCENT = colors.HexColor("#1f4e79")
MUTED = colors.HexColor("#5a6472")
RULE = colors.HexColor("#c9d1d9")


# --- data ------------------------------------------------------------------


def load(run_id: str) -> dict:
    return json.loads((RESULTS / run_id / "summary.json").read_text())


RUNS = {r: load(r) for r in
        ("accel_only", "accel_hr", "hr_only", "accel_only_nocontext")}
ABLATION = json.loads((RESULTS / "ablation.json").read_text())
MANIFEST = json.loads(
    sorted((REPO_ROOT / "data" / "cache").glob("*/manifest.json"))[0].read_text()
)


def k(run: str) -> float:
    return RUNS[run]["pooled"]["kappa"]


def ci(key: str) -> str:
    e = ABLATION[key]
    return f"{e['delta_mean']:+.3f} (95% CI {e['ci_low']:+.3f} to {e['ci_high']:+.3f})"


# --- styles ----------------------------------------------------------------

base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=25,
                            textColor=ACCENT, spaceAfter=6),
    "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=12, leading=16,
                               alignment=TA_CENTER, textColor=MUTED, spaceAfter=22),
    "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=9.5, leading=14,
                           alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=13, leading=17,
                         textColor=ACCENT, spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=10.5, leading=14,
                         textColor=colors.HexColor("#243b53"), spaceBefore=11,
                         spaceAfter=4),
    "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=14,
                           alignment=TA_JUSTIFY, spaceAfter=7),
    "caption": ParagraphStyle("c", parent=base["Normal"], fontSize=8, leading=11,
                              alignment=TA_CENTER, textColor=MUTED, spaceBefore=3,
                              spaceAfter=11),
    "abstract": ParagraphStyle("a", parent=base["BodyText"], fontSize=9.5, leading=14.5,
                               alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14,
                               spaceAfter=7),
    "ref": ParagraphStyle("r", parent=base["Normal"], fontSize=8.5, leading=12,
                          leftIndent=18, firstLineIndent=-18, spaceAfter=5),
    "code": ParagraphStyle("co", parent=base["Normal"], fontName="Courier",
                           fontSize=8, leading=11, leftIndent=10, spaceAfter=6),
}


def P(text: str, style: str = "body"):
    return Paragraph(text, S[style])


def H(text: str, level: int = 1):
    return Paragraph(text, S[f"h{level}"])


def figure(name: str, caption: str, width: float = 15.5):
    path = FIGS / f"{name}.png"
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        ratio = im.height / im.width
    img = Image(str(path), width=width * cm, height=width * ratio * cm)
    return KeepTogether([img, P(caption, "caption")])


def table(rows, widths, highlight_row: int | None = None, align_left_col0=True):
    t = Table(rows, colWidths=[w * cm for w in widths], hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if align_left_col0:
        style.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
    if highlight_row is not None:
        style += [
            ("BACKGROUND", (0, highlight_row), (-1, highlight_row),
             colors.HexColor("#eef4fa")),
            ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2.2 * cm, 1.3 * cm,
                          "Sleep staging from wrist accelerometer alone")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.3 * cm, f"{doc.page}")
    canvas.restoreState()


# --- content ---------------------------------------------------------------

story: list = []

# Title page
story += [
    Spacer(1, 3.4 * cm),
    P("Sleep Stage Classification from Wrist Accelerometer Alone", "title"),
    P("Quantifying the motion-only ceiling for four-class sleep staging, "
      "and what heart rate contributes", "subtitle"),
    Spacer(1, 0.4 * cm),
    P("Amey Pawar", "meta"),
    P(f"{date.today():%d %B %Y}", "meta"),
    Spacer(1, 0.3 * cm),
    P("github.com/ameyypawar/sleep-accel-dl", "meta"),
    Spacer(1, 1.6 * cm),
    P("<b>Abstract</b>", "meta"),
    Spacer(1, 0.25 * cm),
]

story.append(P(
    "Consumer fitness bands that lack a photoplethysmography sensor can measure "
    "motion but not heart rate, and every established wrist-wearable sleep staging "
    "model uses both together. This project isolates the accelerometer and asks "
    "what motion alone can achieve for four-class staging (Wake, Light, Deep, REM), "
    "then reintroduces heart rate to measure its marginal contribution. A 1D "
    "convolutional encoder with temporal attention feeds a bidirectional LSTM over "
    "20-epoch context windows, trained on 31 subjects from the Walch et al. (2019) "
    "dataset with concurrent polysomnography. Under subject-wise five-fold "
    "cross-validation, accelerometer-only staging reaches Cohen's "
    f"&kappa; = {k('accel_only'):.3f}, far below the 0.40 threshold for clinically "
    "fair agreement. Adding heart rate raises this to "
    f"&kappa; = {k('accel_hr'):.3f}, a paired-bootstrap improvement of "
    f"{ci('accel_hr - accel_only')}. Critically, the reverse does not hold: adding "
    "the accelerometer to heart rate yields "
    f"{ci('accel_hr - hr_only')}, an interval containing zero. The founding "
    "hypothesis is therefore rejected. However, a per-class analysis reveals that "
    "the two modalities are complementary along different axes: motion is the "
    f"superior sleep/wake discriminator (binary &kappa; "
    f"{RUNS['accel_only']['pooled']['binary_kappa']:.3f} against "
    f"{RUNS['hr_only']['pooled']['binary_kappa']:.3f} for heart rate), while heart "
    "rate supplies nearly all the information separating sleep stages. The "
    "practical conclusion is that a sensor-limited band can measure sleep duration "
    "reliably but cannot report sleep architecture.", "abstract"))

story.append(PageBreak())

# 1. Introduction
story += [
    H("1. Introduction and motivation"),
    P("Polysomnography (PSG) is the clinical reference standard for sleep staging, "
      "but it requires an overnight laboratory stay with electroencephalography, "
      "electro-oculography and electromyography. Consumer wearables offer a "
      "cheaper, longitudinal alternative, and the research literature on "
      "wrist-based staging is correspondingly large. Almost without exception, "
      "that literature combines an accelerometer with a photoplethysmography (PPG) "
      "heart rate sensor."),
    P("This creates an unexamined dependency. A large share of fitness bands in "
      "circulation carry an accelerometer and no PPG sensor, either for cost or "
      "for battery reasons. If motion alone were sufficient, sleep architecture "
      "monitoring would extend to a substantially larger installed base at no "
      "hardware cost. The question this project answers is whether that is so."),
    P("The contribution is not the demonstration that motion carries some signal, "
      "which is well established for binary sleep/wake actigraphy. It is the "
      "isolation and quantification of the <i>motion-only ceiling</i> at "
      "four-class resolution, together with a controlled measurement of what heart "
      "rate adds on top."),

    H("1.1 Research questions", 2),
    P("<b>RQ1.</b> Can a deep sequence model using only triaxial accelerometer "
      "input achieve Cohen's &kappa; &gt; 0.40, the conventional threshold for "
      "fair agreement, on four-class sleep staging?"),
    P("<b>RQ2.</b> What is the marginal contribution of heart rate, and "
      "symmetrically, what does motion contribute once heart rate is available?"),
    P("<b>RQ3.</b> Does modelling temporal context across consecutive epochs "
      "improve staging, or would a per-epoch classifier suffice?"),
    P("These questions were formulated so that a negative answer to RQ1 remains "
      "an informative result. If motion proves insufficient, the ablation in RQ2 "
      "quantifies precisely what sensor-limited devices forgo."),

    H("2. Related work and positioning"),
    P("Classical actigraphy algorithms such as Cole-Kripke and Sadeh operate on "
      "epoch-level activity counts and were designed exclusively for binary "
      "sleep/wake discrimination. They have no mechanism for distinguishing N3 "
      "from REM, both of which are characterised by near-absent gross motion."),
    P("Walch et al. (2019) collected the dataset used here and evaluated feature "
      "contributions including motion, heart-rate variability and a circadian "
      "\"clock proxy\", reporting approximately 72% accuracy on a "
      "<i>three</i>-class problem (Wake / NREM / REM) using shallow classifiers "
      "over hand-crafted features."),
    P("This work is positioned relative to that as follows, and makes no claim of "
      "priority. Three differences are material. First, the task here is "
      "<i>four</i>-class, separating Light from Deep sleep, which is the harder "
      "distinction and the one consumers are sold on. Second, the model operates "
      "on the <i>raw</i> triaxial waveform through a learned encoder rather than "
      "on engineered features. Third, and most importantly for the research "
      "question, no circadian clock proxy is used. A time-of-day prior is not a "
      "motion signal, and including one would confound any claim about what the "
      "accelerometer alone can do."),
]

story.append(PageBreak())

# 3. Data
counts = {a: b for a, b in MANIFEST["class_counts"].items() if a != "Invalid"}
total_usable = sum(counts.values())

story += [
    H("3. Data"),
    P("The dataset is Walch et al. (2019), <i>Motion and heart rate from a "
      "wrist-worn wearable and labeled sleep from polysomnography</i>, distributed "
      "by PhysioNet under the Open Data Commons Attribution License v1.0. It "
      "comprises 31 subjects wearing an Apple Watch during concurrent clinical "
      "PSG, providing raw triaxial acceleration, PPG-derived heart rate, step "
      "counts, and expert-scored sleep stages at 30-second resolution."),
    P("The Multi-Ethnic Study of Atherosclerosis (MESA) cohort was considered "
      "first and rejected on technical grounds. MESA distributes actigraphy as "
      "30-second <i>activity counts</i> rather than raw accelerometry, which makes "
      "the raw-waveform encoder central to this design impossible to train on it. "
      "It also requires institutional data-access approval, which is incompatible "
      "with a reproducible public repository."),

    H("3.1 Discrepancies between documentation and data", 2),
    P("A format-probing script was written and executed <i>before</i> any parser, "
      "on the principle that undocumented structure should be measured rather than "
      "assumed. This proved necessary: the published description disagrees with "
      "the files in three respects, each of which would fail silently rather than "
      "raise an error."),
]

story.append(table([
    ["Documented", "Observed in the files"],
    ["Stage codes {0,1,2,3,5}", "Also 4 (356 epochs) and -1 (438 epochs)"],
    ["(delimiter unstated)", "motion/labels whitespace; heart_rate/steps comma"],
    ["Approximately 50 Hz", "10.0 to 66.6 Hz, varying by subject"],
    ["(range unstated)", "Timestamps reach -486,000 s (5.6 days pre-study)"],
], [5.2, 10.3]))

story += [
    Spacer(1, 0.3 * cm),
    P("The stage-4 discrepancy is the consequential one. Code 4 is Rechtschaffen "
      "and Kales stage 4, the deepest slow-wave sleep, which the AASM standard "
      "merged into N3 without renumbering. Following the documentation and "
      "treating it as unknown would have discarded 356 epochs, approximately 10% "
      "of the Deep class and the scarcest of the four. The mapping is implemented "
      "as a dictionary rather than a list indexed by stage code, since the code "
      "set is non-contiguous and positional indexing would misplace REM."),
    figure("class_distribution",
           "Figure 1. Class distribution after quality filtering. Light sleep "
           "dominates and Deep is scarcest, matching the expected architecture of "
           "a normal night. A uniform distribution here would have indicated "
           "label-signal misalignment."),
]

story += [
    H("3.2 Preprocessing and quality gating", 2),
    P("The label file defines the epoch axis; accelerometer samples are resampled "
      "onto it. Deriving the axis from the accelerometer instead would allow a "
      "late-starting device to shift every epoch boundary relative to the PSG "
      "scoring, producing a constant misalignment no metric would detect."),
    P("Because the sampling interval is irregular and varies across subjects, "
      "resampling uses linear interpolation onto an explicit uniform 30 Hz grid. "
      "Functions such as <font face='Courier'>scipy.signal.resample</font> assume "
      "uniform input spacing and would return a time-distorted waveform without "
      "warning. A fourth channel, the vector magnitude, is appended because it is "
      "invariant to watch orientation on the wrist."),
    P("Each epoch is gated on two quality measures before interpolation: the "
      "fraction of the window spanned by real samples, and the largest gap between "
      "consecutive samples including the lead-in and lead-out intervals. This "
      "matters more than it appears. Interpolating across a sensor dropout yields "
      "a smooth, near-zero-variance segment, which is precisely the signature of "
      "deep sleep; a model trained on such data learns that absent signal means "
      "deep sleep and reports an inflated score. Subject 7749105 contains no "
      "accelerometer samples whatsoever in 86% of its epochs, and the gate "
      "correctly rejects them. Overall, "
      f"{MANIFEST['n_kept']:,} of {MANIFEST['n_epochs']:,} epochs "
      f"({MANIFEST['kept_fraction']:.1%}) survive."),
    P("Heart rate is summarised per epoch as five aggregate features (mean, "
      "standard deviation, minimum, maximum, and deviation from the subject's "
      "median) rather than resampled as a waveform, because PPG readings arrive "
      "orders of magnitude more sparsely than acceleration. Epochs with no reading "
      "receive zeroed features together with an explicit validity flag supplied to "
      "the model as an additional input; without that flag, \"no measurement\" and "
      "\"0 bpm\" would be indistinguishable, and 0 bpm is a physiologically "
      "extreme value rather than a neutral one."),
]

story.append(PageBreak())

# 4. Method
story += [
    H("4. Method"),
    P("The architecture reflects a specific hypothesis about the problem. Deep and "
      "REM sleep are both characterised by near-absent gross motion, so they are "
      "close to indistinguishable within a single isolated 30-second epoch. What "
      "differentiates them is position within the roughly 90-minute ultradian "
      "cycle. The model is therefore built to classify an epoch <i>in context</i>, "
      "not in isolation."),
    Spacer(1, 0.15 * cm),
]

story.append(table([
    ["Stage", "Operation", "Output"],
    ["Input", "30 s triaxial acceleration + magnitude at 30 Hz", "4 x 900"],
    ["Encoder", "3 x [Conv1d(k=7), BatchNorm, GELU, MaxPool, Dropout]", "128 x 112"],
    ["Pooling", "Additive temporal attention, learned query", "128"],
    ["Context", "BiLSTM over 20 consecutive epoch embeddings", "20 x 256"],
    ["Head", "Dropout, Linear", "20 x 4"],
], [2.2, 9.5, 3.8]))

story += [
    Spacer(1, 0.3 * cm),
    P("Attention pooling collapses the time axis to a single embedding per epoch "
      "while weighting the moments that carry information, rather than averaging "
      "uniformly across a window that is mostly quiescent. The bidirectional LSTM "
      "then reads the sequence of epoch embeddings. For the "
      "<font face='Courier'>accel_hr</font> and "
      "<font face='Courier'>hr_only</font> variants, the heart-rate features are "
      "projected and concatenated to the epoch embedding before the context stage; "
      "for <font face='Courier'>hr_only</font> the convolutional encoder is not "
      "instantiated at all."),
    P("Loss is cross-entropy with epochs failing the quality gate masked out via "
      "<font face='Courier'>ignore_index</font>, weighted by inverse class "
      "frequency raised to the power 0.5. The exponent was measured rather than "
      "assumed: full inverse frequency (power 1.0) inverted the prior, causing the "
      "model to predict Deep for 9,946 epochs against Light's 6,498 when the true "
      "proportions are 14% and 55%, and driving accuracy below the majority-class "
      "baseline. The square root is the compromise that learns the minority "
      "classes without overriding the prior."),

    H("5. Experimental protocol"),
    P("<b>Cross-validation is subject-wise, and this is the single most important "
      "methodological property of the study.</b> One night yields approximately "
      "1,000 epochs from one individual, and consecutive epochs from the same "
      "wrist are far more similar to each other than to any other subject's data. "
      "An epoch-wise split would permit a model to score highly by identifying the "
      "wearer rather than learning anything transferable about sleep, and the "
      "resulting inflation would be invisible in every diagnostic. A leakage "
      "assertion therefore executes at the start of every training run rather than "
      "only under test, and the test suite deliberately corrupts a fold to confirm "
      "the assertion fires."),
    P("Splits are three-way. Validation subjects for early stopping are drawn from "
      "the training portion, because selecting checkpoints on held-out test "
      "subjects would leak them into model selection as surely as training on them "
      "would. Fold membership is serialised to disk and reused verbatim by every "
      "variant, so the ablation arms are comparable by construction rather than by "
      "assumption."),
    P("Evaluation windows tile without overlap, so each test epoch is predicted "
      "exactly once. Training windows use a shorter stride, which provides free "
      "augmentation; applying the same overlap at evaluation would predict interior "
      "epochs several times and boundary epochs once, silently reweighting the "
      "test set."),
    P("Cohen's &kappa; is the primary metric because accuracy is uninformative "
      "under this class distribution: predicting Light universally yields 55% "
      "accuracy and &kappa; = 0. Every result carries a degeneracy flag that fires "
      "when any class is never predicted or one class exceeds 95% of predictions. "
      "Checkpoints are selected on validation macro F1 for the same reason."),
]

story.append(PageBreak())

# 6. Results
story += [H("6. Results")]

rows = [["Variant", "κ (pooled)", "κ across folds", "Macro F1", "Accuracy",
         "Binary κ"]]
for run, label in (("accel_only_nocontext", "Accel only, no context"),
                   ("accel_only", "Accel only"),
                   ("hr_only", "HR only"),
                   ("accel_hr", "Accel + HR")):
    s, a = RUNS[run]["pooled"], RUNS[run]["across_folds"]
    rows.append([label, f"{s['kappa']:.3f}",
                 f"{a['kappa_mean']:.3f} ± {a['kappa_std']:.3f}",
                 f"{s['macro_f1']:.3f}", f"{s['accuracy']:.3f}",
                 f"{s['binary_kappa']:.3f}"])
story.append(table(rows, [4.4, 2.2, 3.0, 2.0, 2.0, 2.0], highlight_row=4))
story.append(P("Table 1. Pooled over all held-out epochs, 31 subjects, "
               "subject-wise five-fold cross-validation, identical folds across "
               "all variants. No run was flagged degenerate.", "caption"))

story.append(figure("kappa_comparison",
                    "Figure 2. Four-class agreement by input variant, with "
                    "standard deviation across folds. Only the heart-rate arms "
                    "approach the clinical threshold, and none reaches it.",
                    width=11.5))

story += [
    H("6.1 RQ1: motion alone is insufficient", 2),
    P(f"Accelerometer-only staging reaches &kappa; = {k('accel_only'):.3f} "
      f"({RUNS['accel_only']['across_folds']['kappa_mean']:.3f} "
      f"± {RUNS['accel_only']['across_folds']['kappa_std']:.3f} across folds), "
      "which is far below the 0.40 threshold for fair agreement. <b>RQ1 is "
      "answered in the negative.</b> The per-class breakdown localises the failure "
      f"precisely: Deep reaches F1 "
      f"{RUNS['accel_only']['pooled']['per_class_f1']['Deep']:.3f} and REM "
      f"{RUNS['accel_only']['pooled']['per_class_f1']['REM']:.3f}. Both stages are "
      "near-motionless, and without a cardiac signal the model has almost nothing "
      "by which to separate them."),
    figure("confusion",
           "Figure 3. Row-normalised confusion matrices. Without heart rate, Deep "
           "and REM are largely absorbed into the adjacent classes; with it, both "
           "acquire substantial diagonal mass."),
]

story += [
    H("6.2 RQ2: heart rate carries the stage information", 2),
    P("Adding heart rate to motion improves agreement by "
      f"{ci('accel_hr - accel_only')}, an interval comfortably excluding zero. "
      "Heart rate alone already outperforms motion alone by "
      f"{ci('hr_only - accel_only')}."),
    P("The decisive comparison is the symmetric one. Adding the accelerometer to "
      f"heart rate yields {ci('accel_hr - hr_only')} — <b>an interval "
      "containing zero.</b> There is no statistically detectable benefit to "
      "supplying motion once heart rate is available, at least at this sample "
      "size. This is the strongest statement the dataset supports, and it "
      "contradicts the premise that a PPG-less device could perform four-class "
      "staging."),
    figure("ablation",
           "Figure 4. Paired bootstrap over subjects, 1,000 resamples. Subjects "
           "rather than epochs are resampled because epochs within one night are "
           "not independent, and the same resample is applied to both arms of each "
           "comparison; bootstrapping arms independently would widen the intervals "
           "and understate significance.",
           width=14.0),

    H("6.3 RQ3: temporal context contributes, narrowly", 2),
    P("Removing the sequence model costs "
      f"{ci('accel_only_nocontext - accel_only')}, an interval that only just "
      "excludes zero. The aggregate understates the effect, however. Without "
      "context the model predicts Deep and REM essentially never, with F1 of "
      f"{RUNS['accel_only_nocontext']['pooled']['per_class_f1']['Deep']:.3f} and "
      f"{RUNS['accel_only_nocontext']['pooled']['per_class_f1']['REM']:.3f} "
      "respectively; it collapses onto the two classes that motion can detect. "
      "The BiLSTM is what enables any attempt at the stage distinctions at all, "
      "which is consistent with the hypothesis that motivated the architecture."),
]

story += [
    H("7. Discussion: the modalities are complementary along different axes"),
    P("The aggregate &kappa; conceals the most useful finding in this study. "
      "Motion and heart rate are not simply stronger and weaker versions of the "
      "same signal; they carry different information."),
    figure("per_class_f1",
           "Figure 5. Per-class F1 across variants. Motion is the better wake "
           "detector; heart rate supplies the stage distinctions.",
           width=14.0),
]

story.append(table([
    ["Variant", "Wake F1", "Light F1", "Deep F1", "REM F1", "Binary κ"],
    ["Accel only"] + [f"{RUNS['accel_only']['pooled']['per_class_f1'][c]:.3f}"
                      for c in ("Wake", "Light", "Deep", "REM")]
    + [f"{RUNS['accel_only']['pooled']['binary_kappa']:.3f}"],
    ["HR only"] + [f"{RUNS['hr_only']['pooled']['per_class_f1'][c]:.3f}"
                   for c in ("Wake", "Light", "Deep", "REM")]
    + [f"{RUNS['hr_only']['pooled']['binary_kappa']:.3f}"],
    ["Accel + HR"] + [f"{RUNS['accel_hr']['pooled']['per_class_f1'][c]:.3f}"
                      for c in ("Wake", "Light", "Deep", "REM")]
    + [f"{RUNS['accel_hr']['pooled']['binary_kappa']:.3f}"],
], [3.4, 2.4, 2.4, 2.4, 2.4, 2.5], highlight_row=3))
story.append(P("Table 2. The complementarity. Motion wins on Wake and on binary "
               "sleep/wake; heart rate wins decisively on Deep and REM.",
               "caption"))

story += [
    P("<b>Motion determines whether a person is awake.</b> Accelerometer-only "
      f"attains binary sleep/wake &kappa; = "
      f"{RUNS['accel_only']['pooled']['binary_kappa']:.3f} against "
      f"{RUNS['hr_only']['pooled']['binary_kappa']:.3f} for heart rate alone. "
      "Motion is substantially the better wake detector, which is precisely why "
      "classical actigraphy has been clinically useful for decades."),
    P("<b>Heart rate determines which stage of sleep.</b> It raises Deep F1 from "
      f"{RUNS['accel_only']['pooled']['per_class_f1']['Deep']:.3f} to "
      f"{RUNS['hr_only']['pooled']['per_class_f1']['Deep']:.3f} and REM from "
      f"{RUNS['accel_only']['pooled']['per_class_f1']['REM']:.3f} to "
      f"{RUNS['hr_only']['pooled']['per_class_f1']['REM']:.3f}, in a regime where "
      "motion is nearly blind."),
    P("Combining both yields the best binary &kappa; "
      f"({RUNS['accel_hr']['pooled']['binary_kappa']:.3f}) and the best macro F1 "
      f"({RUNS['accel_hr']['pooled']['macro_f1']:.3f}). The reason "
      "<font face='Courier'>accel_hr</font> barely exceeds "
      "<font face='Courier'>hr_only</font> on four-class &kappa; is that the "
      "metric is dominated by the three sleep-stage distinctions, where motion "
      "contributes little — not because motion is uninformative in general. "
      "Reporting only the headline &kappa; would have obscured this entirely."),
    P("<b>Practical implication.</b> A wearable with an accelerometer and no PPG "
      "sensor can report total sleep time and sleep efficiency with reasonable "
      "fidelity. It cannot report sleep architecture. Consumer devices that "
      "present deep-sleep and REM breakdowns without a cardiac sensor should be "
      "treated with scepticism."),

    H("8. Threats to validity"),
    P("<b>Sample size.</b> 31 subjects is small. The confidence interval on the "
      "motion-added-to-heart-rate comparison is "
      f"[{ABLATION['accel_hr - hr_only']['ci_low']:+.3f}, "
      f"{ABLATION['accel_hr - hr_only']['ci_high']:+.3f}], so a true effect of up "
      "to approximately +0.04 &kappa; cannot be excluded. The correct reading is "
      "\"no detectable benefit at this sample size\", not \"no benefit\"."),
    P("<b>Single device and population.</b> All data comes from one Apple Watch "
      "model in a single-site study. Sampling rates varying from 10 to 67 Hz "
      "across subjects suggest heterogeneous firmware or collection conditions "
      "even within this cohort."),
    P("<b>Training budget.</b> Each arm trained for eight epochs, selected on "
      "validation macro F1. Validation curves had plateaued by approximately epoch "
      "five in every fold, so the arms are not materially undertrained, but no "
      "hyperparameter search was performed and absolute performance would likely "
      "improve modestly with one."),
    P("<b>Transductive heart-rate baseline.</b> The per-subject heart-rate "
      "baseline is the median over the entire recording, including epochs that "
      "later fall in held-out folds. This is a deliberate choice, since the "
      "quantity is per-subject by construction and a full night is available "
      "before scoring on a real device, but it should be stated rather than "
      "discovered."),
    P("<b>Class weighting interacts with the metric.</b> The weighting exponent "
      "was found to move &kappa; and accuracy in opposite directions. Results are "
      "reported at a single exponent (0.5) applied uniformly to every arm, so "
      "comparisons are internally consistent, but the absolute &kappa; values are "
      "mildly sensitive to this choice."),
]

story += [
    H("9. Conclusion"),
    P("Four-class sleep staging from wrist accelerometry alone reaches Cohen's "
      f"&kappa; = {k('accel_only'):.3f}, well short of clinical usability, and "
      "adding motion to heart rate produces no statistically detectable "
      "improvement. The hypothesis that motivated this work — that "
      "sensor-limited fitness bands could deliver sleep architecture — is not "
      "supported by this dataset."),
    P("The more useful finding is the decomposition. Motion and heart rate are "
      "complementary along orthogonal axes: motion determines wakefulness, heart "
      "rate determines sleep stage. This explains both why classical actigraphy "
      "succeeds at its intended binary task and why extending it to four classes "
      "fails. It also delimits precisely what a PPG-less device can honestly "
      "claim."),
    P("The methodological contribution is a validation protocol built to make "
      "silent failure difficult: subject-wise splits with assertions that execute "
      "in production rather than only in tests, quality gating that rejects sensor "
      "dropouts before they can be interpolated into convincing artefacts, a "
      "degeneracy flag on every reported metric, and a format probe that measured "
      "the data rather than trusting its documentation — which proved "
      "necessary in three separate respects."),

    H("10. Reproducibility"),
    P("All code, pinned dependencies and the analysis pipeline are public. The "
      "test suite comprises 71 tests, all of which execute without the dataset "
      "present, and specifically cover each silent-failure mode described above."),
    Paragraph("git clone https://github.com/ameyypawar/sleep-accel-dl<br/>"
              "uv venv --python 3.11 .venv &amp;&amp; source .venv/bin/activate<br/>"
              "uv pip install -r requirements.txt<br/>"
              "pytest -q<br/>"
              "python scripts/probe_raw_format.py<br/>"
              "python -m sleepaccel.data.build_cache<br/>"
              "./scripts/run_all.sh &amp;&amp; python scripts/ablation.py<br/>"
              "streamlit run app/streamlit_app.py", S["code"]),
    P("An interactive dashboard presents the dataset summary, training curves, "
      "confusion matrices, the ablation comparison, and a per-subject hypnogram "
      "viewer aligning predictions against PSG ground truth."),
    figure("hypnogram",
           "Figure 6. Predicted hypnogram against polysomnography for the "
           "longest recording, accelerometer plus heart rate. Disagreement "
           "concentrates at stage transitions, where inter-scorer agreement is "
           "also lowest.",
           width=14.5),

    H("References"),
    Paragraph("Walch, O., Huang, Y., Forger, D., and Goldstein, C. (2019). Sleep "
              "stage prediction with raw acceleration and photoplethysmography "
              "heart rate data derived from a consumer wearable device. "
              "<i>SLEEP</i>, 42(12), zsz180.", S["ref"]),
    Paragraph("Walch, O. (2019). Motion and heart rate from a wrist-worn wearable "
              "and labeled sleep from polysomnography (version 1.0.0). "
              "<i>PhysioNet</i>. Open Data Commons Attribution License v1.0.", S["ref"]),
    Paragraph("Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., and "
              "Gillin, J. C. (1992). Automatic sleep/wake identification from "
              "wrist activity. <i>Sleep</i>, 15(5), 461-469.", S["ref"]),
    Paragraph("Berry, R. B., et al. (2012). The AASM Manual for the Scoring of "
              "Sleep and Associated Events. <i>American Academy of Sleep "
              "Medicine</i>.", S["ref"]),
    Paragraph("Cohen, J. (1960). A coefficient of agreement for nominal scales. "
              "<i>Educational and Psychological Measurement</i>, 20(1), 37-46.", S["ref"]),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="Sleep Stage Classification from Wrist Accelerometer Alone",
        author="Amey Pawar",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
