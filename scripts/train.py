"""Train and cross-validate. One run per input variant.

    python scripts/train.py --variant accel_only --run-id rq1
    python scripts/train.py --variant accel_hr  --run-id rq2_accel_hr
    python scripts/train.py --variant hr_only   --run-id rq2_hr_only

Every run reuses the same folds.json, so the ablation arms are comparable by
construction rather than by hope.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import time

import numpy as np
import torch

from sleepaccel.data.dataset import (
    SleepWindows,
    class_counts,
    fit_hr_normalizer,
    fit_normalizer,
    make_loader,
)
from sleepaccel.data.labels import INVALID
from sleepaccel.data.splits import (
    assert_no_subject_leakage,
    load_folds,
    save_folds,
    subject_folds,
)
from sleepaccel.metrics import aggregate, compute_metrics
from sleepaccel.models import (
    SleepStagingNet,
    make_class_weights,
    masked_cross_entropy,
    select_device,
)
from sleepaccel.paths import REPO_ROOT
from sleepaccel.seed import seed_everything


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) over all unmasked epochs."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["waveform"].to(device), batch["hr"].to(device))
            pred = logits.argmax(-1).cpu().numpy().ravel()
            true = batch["labels"].numpy().ravel()
            keep = true != INVALID
            trues.append(true[keep])
            preds.append(pred[keep])
    if not trues:
        return np.array([]), np.array([])
    return np.concatenate(trues), np.concatenate(preds)


def evaluate_per_subject(model, subjects, cache, device, args, norm, hr_norm):
    """Evaluate each test subject separately so epochs keep their subject tag.

    The paired bootstrap resamples subjects, so every epoch must know which
    subject it came from. Evaluating the fold as one pooled loader would lose
    that.
    """
    trues, preds, tags = [], [], []
    for sid in subjects:
        ds = SleepWindows(
            [sid],
            cache_dir=cache,
            context_len=args.context_len,
            stride=args.context_len,
            input_variant=args.variant,
            normalizer=norm,
            hr_normalizer=hr_norm,
        )
        y_true, y_pred = evaluate(model, make_loader(ds, args.batch_size, False), device)
        trues.append(y_true)
        preds.append(y_pred)
        tags.append(np.full(y_true.size, sid, dtype=object))
    return (
        np.concatenate(trues) if trues else np.array([]),
        np.concatenate(preds) if preds else np.array([]),
        np.concatenate(tags) if tags else np.array([], dtype=object),
    )


def train_fold(fold, cache, device, args):
    seed_everything(args.seed + fold.index)

    norm = fit_normalizer(list(fold.train_subjects), cache)
    hr_norm = fit_hr_normalizer(list(fold.train_subjects), cache)
    common = dict(
        cache_dir=cache,
        context_len=args.context_len,
        input_variant=args.variant,
        normalizer=norm,
        hr_normalizer=hr_norm,
    )

    train_ds = SleepWindows(list(fold.train_subjects), stride=args.train_stride, **common)
    val_ds = SleepWindows(list(fold.val_subjects), stride=args.context_len, **common)

    train_dl = make_loader(train_ds, args.batch_size, shuffle=True)
    val_dl = make_loader(val_ds, args.batch_size, shuffle=False)

    model = SleepStagingNet(
        input_variant=args.variant,
        context_model=args.context_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    weights = make_class_weights(class_counts(list(fold.train_subjects), cache))
    weight_tensor = torch.tensor(weights, device=device) if weights is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(batch["waveform"].to(device), batch["hr"].to(device))
            loss = masked_cross_entropy(logits, batch["labels"].to(device), weight_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(batch["labels"])
            seen += len(batch["labels"])
        sched.step()

        y_true, y_pred = evaluate(model, val_dl, device)
        val = compute_metrics(y_true, y_pred)
        history.append(
            {"epoch": epoch, "train_loss": total / max(seen, 1),
             "val_macro_f1": val["macro_f1"], "val_kappa": val["kappa"]}
        )
        # Selection on macro F1, not accuracy: accuracy is gamed by the
        # majority class, and this dataset is 55% Light.
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"    epoch {epoch:2d}/{args.epochs}  loss={total/max(seen,1):.4f}  "
            f"val_f1={val['macro_f1']:.4f}  val_kappa={val['kappa']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, tags = evaluate_per_subject(
        model, list(fold.test_subjects), cache, device, args, norm, hr_norm
    )
    result = compute_metrics(y_true, y_pred)
    result["fold"] = fold.index
    result["test_subjects"] = list(fold.test_subjects)
    result["history"] = history
    return result, y_true, y_pred, tags


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="accel_only",
                   choices=["accel_only", "accel_hr", "hr_only"])
    p.add_argument("--context-model", default="bilstm", choices=["bilstm", "none"])
    p.add_argument("--run-id", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--folds", default="results/folds.json")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--context-len", type=int, default=20)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    run_id = args.run_id or f"{args.variant}_{args.context_model}"
    device = select_device(args.device)
    print(f"run {run_id} | variant={args.variant} | context={args.context_model} | device={device}")

    cache = _Path(args.cache) if args.cache else sorted((REPO_ROOT / "data" / "cache").glob("*/"))[0]
    subject_ids = sorted(p.stem for p in cache.glob("*.npz"))
    print(f"cache {cache.name} | {len(subject_ids)} subjects")

    folds_path = _Path(args.folds)
    if folds_path.exists():
        folds = load_folds(folds_path)
        print(f"reusing folds from {folds_path}")
    else:
        folds = subject_folds(subject_ids, "grouped_5fold", args.seed, args.n_folds)
        save_folds(folds, folds_path)
        print(f"wrote new folds to {folds_path}")
    assert_no_subject_leakage(folds)

    out = REPO_ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)

    fold_metrics, all_true, all_pred, all_subj = [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        print(f"\n  fold {fold.index}: test={list(fold.test_subjects)}")
        result, y_true, y_pred, tags = train_fold(fold, cache, device, args)
        fold_metrics.append(result)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_subj.append(tags)
        print(f"  -> fold {fold.index} kappa={result['kappa']:.4f} "
              f"macro_f1={result['macro_f1']:.4f} degenerate={result['is_degenerate']}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    pooled = compute_metrics(y_true, y_pred)
    summary = {
        "run_id": run_id,
        "variant": args.variant,
        "context_model": args.context_model,
        "pooled": pooled,
        "across_folds": aggregate(fold_metrics),
        "folds": fold_metrics,
        "runtime_s": time.perf_counter() - start,
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", y_true=y_true, y_pred=y_pred,
                        subjects=np.concatenate(all_subj).astype(str))

    print(f"\n=== {run_id} ===")
    print(f"pooled kappa    : {pooled['kappa']:.4f}")
    print(f"pooled macro F1 : {pooled['macro_f1']:.4f}")
    print(f"accuracy        : {pooled['accuracy']:.4f}")
    print(f"per-class F1    : {pooled['per_class_f1']}")
    print(f"degenerate      : {pooled['is_degenerate']}")
    print(f"across folds    : kappa {summary['across_folds']['kappa_mean']:.4f} "
          f"+/- {summary['across_folds']['kappa_std']:.4f}")
    print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
