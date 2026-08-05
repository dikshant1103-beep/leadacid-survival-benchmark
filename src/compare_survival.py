#!/usr/bin/env python
"""
Rigorous algorithm comparison for lead-acid failure prognostics.

    python compare_survival.py --horizons 28 84

Seven survival models plus an age-only reference, evaluated with grouped
cross-validation and the metrics the prognostics literature actually reports.

METRICS, and why each is here:

  Harrell's C-index   the familiar one, but optimistically biased when
                      censoring is heavy — and 54 % of our rows are censored.
  Uno's C-index       inverse-probability-of-censoring weighted. This is the
                      number to quote.
  time-dependent AUC  discrimination AT a specific horizon. A model can rank
                      lifetimes well overall and still be useless at four weeks.
  Brier score         calibration, not just ranking. Low Brier means the
                      predicted probabilities mean something.
  binary AUC          computed at each horizon so the result is directly
                      comparable to the 0.806 the earlier logistic classifier
                      reached.

GROUPING: folds are split by BATTERY. Each battery contributes up to seven
landmark rows, and all of them move together — otherwise day-180 and day-420 of
the same cell land on opposite sides and every score is inflated.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent


def build_models(seed: int, n_features: int) -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import (ExtraSurvivalTrees,
                                 GradientBoostingSurvivalAnalysis,
                                 RandomSurvivalForest)
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

    return {
        "Cox PH (age only)": ("age", make_pipeline(
            StandardScaler(), CoxPHSurvivalAnalysis(alpha=1e-3))),
        "Cox PH (all features)": ("all", make_pipeline(
            StandardScaler(), CoxPHSurvivalAnalysis(alpha=1.0))),
        "Coxnet (elastic net)": ("all", make_pipeline(
            StandardScaler(),
            CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01,
                                   fit_baseline_model=True))),
        "Random Survival Forest": ("all", RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=seed)),
        "Extra Survival Trees": ("all", ExtraSurvivalTrees(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=seed)),
        "Gradient Boosted Survival": ("all", GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed)),
        "GBS (component-wise LS)": ("all", GradientBoostingSurvivalAnalysis(
            n_estimators=400, learning_rate=0.05, max_depth=2,
            subsample=0.8, random_state=seed)),
    }


def evaluate_fold(model, d, tr, te, horizons):
    """Fit on tr, score on te. Returns metrics or None if the fold is degenerate."""
    from sklearn.metrics import roc_auc_score
    from sksurv.metrics import (brier_score, concordance_index_censored,
                                concordance_index_ipcw, cumulative_dynamic_auc)

    y_tr, y_te = d.y[tr], d.y[te]
    model.fit(d.x[tr], y_tr)
    risk = model.predict(d.x[te])          # higher = higher risk

    out = {}
    out["harrell_c"] = concordance_index_censored(
        d.event[te], d.duration[te], risk)[0]

    # Uno's C needs a truncation time inside the follow-up of both sets
    tau = min(d.duration[tr][d.event[tr]].max(), d.duration[te][d.event[te]].max())
    try:
        out["uno_c"] = concordance_index_ipcw(y_tr, y_te, risk, tau=tau * 0.9)[0]
    except Exception:
        out["uno_c"] = np.nan

    for h in horizons:
        # binary label: did this row fail within h days of its landmark?
        lab = d.event[te] & (d.duration[te] <= h)
        # rows censored BEFORE h have unknown status and must be excluded
        known = d.event[te] | (d.duration[te] > h)
        if lab.sum() >= 5 and (~lab[known]).sum() >= 5:
            out[f"auc_{h}"] = roc_auc_score(lab[known], risk[known])
        else:
            out[f"auc_{h}"] = np.nan

        try:
            td, _ = cumulative_dynamic_auc(y_tr, y_te, risk, np.array([float(h)]))
            out[f"tdauc_{h}"] = float(td[0])
        except Exception:
            out[f"tdauc_{h}"] = np.nan

        try:
            surv = model.predict_survival_function(d.x[te], return_array=False)
            prob = np.array([fn(h) for fn in surv])
            out[f"brier_{h}"] = float(
                brier_score(y_tr, y_te, prob[:, None], np.array([float(h)]))[1][0])
        except Exception:
            out[f"brier_{h}"] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Survival algorithm comparison")
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--horizons", type=int, nargs="+", default=[28, 84])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--out", type=Path, default=HERE / "survival_comparison.json")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold

    from survival_data import build, verify

    print(f"building landmark survival data (window {args.window} d) …")
    d = build(args.daily, window_days=args.window)
    verify(d)

    age_idx = [d.columns.index("age_days")]
    print(f"{d.x.shape[1]} features | horizons {args.horizons} d\n")

    cv = GroupKFold(n_splits=args.folds)
    splits = list(cv.split(d.x, d.event, d.groups))
    # confirm the grouping actually holds
    for tr, te in splits:
        assert not (set(d.groups[tr]) & set(d.groups[te])), "battery leaked across fold"
    print(f"grouped {args.folds}-fold CV — no battery spans a fold\n")

    models = build_models(args.seed, d.x.shape[1])
    results: dict[str, dict] = {}

    hdr = (f"{'model':28s} {'Uno C':>8s} {'Harrell':>8s}"
           + "".join(f" {'AUC@'+str(h):>9s}" for h in args.horizons)
           + "".join(f" {'Brier@'+str(h):>10s}" for h in args.horizons) + f" {'s':>6s}")
    print(hdr)
    print("-" * len(hdr))

    for name, (scope, proto) in models.items():
        t0 = time.time()
        cols = age_idx if scope == "age" else list(range(d.x.shape[1]))
        sub = type(d)(x=d.x[:, cols], duration=d.duration, event=d.event,
                      groups=d.groups, landmark=d.landmark,
                      columns=[d.columns[i] for i in cols])
        fold_metrics = []
        for tr, te in splits:
            try:
                from sklearn.base import clone
                fold_metrics.append(evaluate_fold(clone(proto), sub, tr, te,
                                                  args.horizons))
            except Exception as exc:
                print(f"  {name}: fold failed — {type(exc).__name__}: {exc}")
        if not fold_metrics:
            continue
        agg = {k: (float(np.nanmean([m[k] for m in fold_metrics])),
                   float(np.nanstd([m[k] for m in fold_metrics])))
               for k in fold_metrics[0]}
        results[name] = agg
        row = (f"{name:28s} {agg['uno_c'][0]:8.3f} {agg['harrell_c'][0]:8.3f}"
               + "".join(f" {agg[f'auc_{h}'][0]:9.3f}" for h in args.horizons)
               + "".join(f" {agg[f'brier_{h}'][0]:10.4f}" for h in args.horizons)
               + f" {time.time()-t0:6.1f}")
        print(row)

    # ---- reference points
    print("\n" + "=" * 78)
    print("REFERENCE POINTS")
    print("=" * 78)
    # REMOVED: two fabricated reference points used to print here --
    #     "Voronov & Frisk (Scania fleet, lead-acid, RSF)   AUC 0.69 - 0.772"
    #     "Voronov & Frisk (same study, Cox regression)     AUC 0.63 - 0.675"
    # NEITHER range appears in any of the three Linkoping papers, all read in
    # full. Voronov et al. (IEEE T-Rel 67(2), 2018) report error rate and
    # concordance, no AUC at all, and explicitly state that "a straightforward
    # application of the Cox regression model is not applicable and motivates
    # our choice of the non-parametric RSF model" -- they never fitted Cox, so
    # no RSF-vs-Cox ordering exists in their work. Frisk et al. (PHM 2014) use
    # AUC only as a variable-importance score; Frisk & Krysander (IFAC 2015)
    # report neither.
    #
    # These invented numbers are what the manuscript's claim of "reversing the
    # ordering reported by Voronov et al." was built on. That sentence is
    # unsupported and is being removed at journal submission.
    print("  our earlier logistic classifier, 8-week horizon  AUC 0.806")
    print("  random guessing                                  AUC 0.500")

    best = max(results, key=lambda k: results[k]["uno_c"][0])
    print(f"\nbest by Uno's C-index: {best} ({results[best]['uno_c'][0]:.3f})")
    for h in args.horizons:
        b = max(results, key=lambda k: (results[k][f"auc_{h}"][0]
                                        if np.isfinite(results[k][f"auc_{h}"][0]) else -1))
        print(f"best at {h:3d}-day horizon:  {b} (AUC {results[b][f'auc_{h}'][0]:.3f} "
              f"± {results[b][f'auc_{h}'][1]:.3f})")

    args.out.write_text(json.dumps(
        {"results": results, "horizons": args.horizons, "window_days": args.window,
         "n_rows": len(d), "n_batteries": int(len(np.unique(d.groups))),
         "n_events": int(d.event.sum())}, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
