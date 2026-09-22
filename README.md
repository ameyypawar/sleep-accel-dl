# Sleep staging from wrist accelerometer alone

Four-class sleep staging — **Wake / Light / Deep / REM** — from a wrist-worn
accelerometer, with **no heart rate sensor and no EEG**.

Billions of people wear fitness bands with an accelerometer and no PPG. Every
wrist-wearable staging model of note uses motion *and* heart rate together.
This project strips heart rate out entirely and asks what motion alone can do,
then puts it back to measure exactly what it was worth.

**The answer is that it is worth a great deal.** Motion alone reaches
Cohen's κ = 0.101, well short of the 0.40 clinical threshold; adding heart rate
takes it to 0.346. And once heart rate is present, motion adds nothing
statistically detectable on top (Δκ = +0.018, 95% CI [−0.005, +0.042]). The
hypothesis that a cheap PPG-less band could do four-class staging does not hold
here — but the breakdown by class is more interesting than the headline, and is
in [Results](#results).

```
30s epoch of triaxial acceleration  [4 x 900]   (x, y, z, magnitude @ 30 Hz)
        |
   1D CNN, 3 blocks                              local motion micropatterns
        |
   temporal attention pooling                    which moments matter
        |
   BiLSTM over 20 consecutive epochs             sleep-cycle context
        |
   softmax -> Wake / Light / Deep / REM
```

The sequence model is the load-bearing part. Deep and REM are both
near-motionless inside a single 30-second window; what separates them is where
they sit in the night's 90-minute cycle. There is a `--context-model none` arm
whose only job is to demonstrate that rather than assert it.

## Research questions

**RQ1** — Can motion alone reach Cohen's κ > 0.40, the threshold for clinically
"fair agreement"?

**RQ2** — How much does heart rate add on top? Same architecture, same folds,
three input variants: `accel_only`, `accel_hr`, `hr_only`, compared with a
paired bootstrap over subjects.

**RQ3** — Does the sequence context actually do the work, or would a per-epoch
classifier do as well?

RQ2 is the contribution. Either outcome is worth reporting: if heart rate adds
a lot, that quantifies what cheap bands are losing; if it adds little, cheap
bands are good enough.

## Results

31 subjects, subject-wise 5-fold cross-validation, identical folds across every
arm. Pooled over all held-out epochs.

| Variant | κ | macro F1 | Accuracy | Binary κ |
|---|---|---|---|---|
| `accel_only` | 0.101 | 0.336 | 0.517 | 0.419 |
| `accel_only`, no context | 0.078 | 0.281 | 0.568 | 0.361 |
| `hr_only` | 0.327 | 0.493 | 0.580 | 0.242 |
| **`accel_hr`** | **0.346** | **0.537** | 0.591 | **0.443** |

Paired bootstrap over subjects, 1,000 resamples:

| Comparison | Δκ | 95% CI | |
|---|---|---|---|
| `accel_hr` − `accel_only` | +0.244 | [+0.202, +0.284] | excludes zero |
| `hr_only` − `accel_only` | +0.226 | [+0.180, +0.271] | excludes zero |
| `accel_hr` − `hr_only` | +0.018 | [−0.005, +0.042] | **includes zero** |
| no-context − `accel_only` | −0.023 | [−0.046, −0.000] | excludes zero |

### The hypothesis does not survive

**RQ1: no.** Motion alone reaches κ = 0.101, far below the 0.40 clinical
threshold. Accelerometer-only 4-class staging is not clinically viable on this
data. The per-class F1 shows exactly where it fails: Deep 0.090 and REM 0.105.
Both stages are near-motionless, and without heart rate the model has almost
nothing to separate them by.

**RQ2: heart rate carries the stage information, and motion adds nothing
detectable on top of it.** The `accel_hr` − `hr_only` interval contains zero.
That is the strongest statement this dataset supports, and it contradicts the
premise that a cheap PPG-less band could do this job.

**RQ3: the sequence model earns its place, narrowly.** Removing it costs
κ 0.023, an interval that only just excludes zero. The per-class numbers are
more telling than the aggregate: without context the model predicts Deep and
REM essentially never (F1 0.000 and 0.001). The BiLSTM is what lets it attempt
those classes at all.

### The interesting part: the two signals do different jobs

Aggregate κ hides this, and it is the most useful thing here.

| | Wake F1 | Light F1 | Deep F1 | REM F1 | Binary κ |
|---|---|---|---|---|---|
| `accel_only` | 0.471 | 0.679 | 0.090 | 0.105 | 0.419 |
| `hr_only` | 0.291 | 0.664 | 0.487 | 0.531 | 0.242 |
| `accel_hr` | 0.490 | 0.664 | 0.455 | 0.539 | 0.443 |

**Motion tells you whether someone is awake.** Accelerometer-only reaches
binary κ 0.419 against heart-rate-only's 0.242 — motion is substantially the
better sleep/wake signal, which is why classical actigraphy works.

**Heart rate tells you which stage of sleep.** It lifts Deep from 0.090 to
0.487 and REM from 0.105 to 0.531, where motion is nearly blind.

So the two are complementary along different axes, and combining them gives the
best binary κ (0.443) and the best macro F1 (0.537). The reason `accel_hr`
barely beats `hr_only` on overall κ is that four-class κ is dominated by the
stage distinctions, where motion contributes little — not because motion is
uninformative in general.

A practical reading: a PPG-less band can tell you *how long* you slept, and
does it well. It cannot tell you how much of that was deep or REM sleep.

## Data

[Walch et al. 2019](https://physionet.org/content/sleep-accel/1.0.0/), *Motion
and heart rate from a wrist-worn wearable and labeled sleep from
polysomnography*. 31 subjects, Apple Watch raw triaxial acceleration and PPG
heart rate with concurrent PSG scoring. Open Data Commons Attribution License
v1.0 — no credentialed access.

MESA was the obvious choice and is the wrong one here: it ships 30-second
*activity counts*, not raw accelerometer, so the architecture above cannot be
built on it.

### What the files actually contain

The dataset documentation and the files disagree, in ways that fail silently
rather than loudly. Measured across all 31 subjects:

| Documented | Actual |
|---|---|
| Stage codes `{0,1,2,3,5}` | Also `4` (356 epochs) and `-1` (438 epochs) |
| — | `motion`/`labels` are whitespace-delimited; `heart_rate`/`steps` are **comma**-delimited |
| ~50 Hz | 10.0 – 66.6 Hz, varying by subject |
| — | Timestamps reach **-486,000 s**: 5.6 days of pre-study wear |

Stage 4 is R&K's deepest slow-wave sleep, merged into N3 by AASM. Following the
documentation and treating it as unknown would discard ~10% of the Deep class,
already the scarcest. `scripts/probe_raw_format.py` exists to measure all of
this rather than assume it, and is worth running after any re-download.

Resulting distribution over 27,211 epochs:

| Wake | Light | Deep | REM | Unscored |
|---|---|---|---|---|
| 2,429 (9%) | 14,775 (55%) | 3,685 (14%) | 5,884 (22%) | 438 |

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# ~577 MB. PhysioNet drops connections; -C - resumes rather than restarting.
curl -L -C - -o data/raw/sleep-accel.zip \
  https://physionet.org/static/published-projects/sleep-accel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip
unzip -q data/raw/sleep-accel.zip -d data/raw/
```

## Running it

```bash
pytest -q                                       # 71 tests, no dataset needed
python scripts/probe_raw_format.py              # verify the files match expectations
python -m sleepaccel.data.build_cache           # ~30 s, builds the epoch tensors
./scripts/run_all.sh                            # all four arms, ~40 min on an M-series Mac
python scripts/ablation.py                      # paired bootstrap
streamlit run app/streamlit_app.py              # dashboard
```

## Dashboard

`streamlit run app/streamlit_app.py` — dataset overview and class balance,
RQ1 metrics with confusion matrix, the RQ2 ablation comparison, training
curves, and a per-subject hypnogram showing prediction against PSG across a
night with a disagreement ribbon.

It reads only what the runs wrote to `results/` and imports torch lazily, so it
opens instantly and is unaffected by a training job running alongside it.

## Things that would have silently produced a wrong result

Each of these yields a plausible-looking number rather than an error, and each
has a test.

**Subject leakage.** One night is ~1,000 epochs from one person, and epochs
from the same wrist are far more alike than any two subjects. Split epoch-wise
and the model scores well by recognising whose wrist it is looking at.
`assert_no_subject_leakage` runs at the start of every training run, not only
under pytest, and the tests corrupt a fold deliberately to prove it fires.
Splits are three-way: early stopping on the test subjects leaks them into model
selection just as surely as training on them would.

**Interpolating across sensor dropouts.** `np.interp` over a watch dropout
gives a smooth, near-zero-variance segment — the exact signature of deep sleep.
Subject 7749105 has *no* accelerometer samples in 86% of its epochs; without
the coverage and gap gate those would have become flat lines labelled as
whatever PSG scored, teaching the model that absent signal means sleep.

**Negative timestamps.** `int(t/30)` truncates toward zero, mapping `t=-15` and
`t=+15` both onto epoch 0 and folding pre-study data into the first labelled
epoch. Only floor division is correct.

**The wrong resampler.** `scipy.signal.resample` and `decimate` assume uniform
input spacing. This data is irregular and varies 10–67 Hz by subject, so they
return a time-distorted waveform with no warning.

**Accuracy as a selection metric.** Light is 55% of epochs, so predicting Light
always scores 55% accuracy and κ = 0. Selection is on macro F1, and
`is_degenerate` flags any run where a class is never predicted.

**Unpaired bootstrapping.** Bootstrapping each ablation arm independently gives
an interval wide enough to straddle zero, causing an under-claim. The resample
is shared across arms and drawn over subjects, not epochs.

## Relation to prior work

Walch et al. compared feature contributions on this data, including motion and
heart-rate variability, reaching ~72% on **three** classes (Wake / NREM / REM)
with shallow classifiers on hand-crafted features — plus a "clock proxy", a
circadian time-of-day prior that is not motion.

This is not a priority claim. The differences are: **four** classes rather than
three, splitting Light from Deep; a deep sequence model on the **raw** waveform
rather than engineered features; and **no circadian prior**, which is what makes
a motion-only claim actually clean.

## Layout

```
src/sleepaccel/
  config.py, device.py, seed.py, paths.py    configuration and environment
  data/labels.py                             stage-code mapping
  data/epochs.py                             alignment, resampling, quality gating
  data/hr_features.py                        per-epoch HR aggregates
  data/raw_io.py                             parsers, written against measured files
  data/splits.py                             subject-wise CV
  data/dataset.py                            windowed torch dataset
  data/build_cache.py                        one-time epoch cache
  models.py                                  CNN + attention + BiLSTM
  metrics.py                                 kappa, degeneracy guard, bootstrap
scripts/                                     probe, train, ablation, run_all
app/streamlit_app.py                         dashboard
tests/                                       71 tests, all dataset-free
```

## Licence and attribution

Code is MIT. The dataset is ODC-BY v1.0 and is not redistributed here.

> Walch, O., Huang, Y., Forger, D., Goldstein, C. (2019). Sleep stage
> prediction with raw acceleration and photoplethysmography heart rate data
> derived from a consumer wearable device. *SLEEP*, 42(12).
