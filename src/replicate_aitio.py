#!/usr/bin/env python
"""
Head-to-head against Aitio & Howey (Joule 2021) on THEIR protocol.

    python replicate_aitio.py

Their published claim, from the paper's own wording:

    "using 5-fold stratified cross-validation ... this technique gives 82%
     balanced accuracy of end-of-life failure prediction at the time of failure
     versus a 66% benchmark, and 73% accuracy 8 weeks in advance of failure
     versus a 51% benchmark."

Three things must match for the comparison to mean anything, and none of them
matched in our earlier runs:

  1. BALANCED ACCURACY, (sensitivity + specificity)/2 — not AUC, not raw
     accuracy. On a 9% base rate raw accuracy is 91% for free.

  2. FAILURE-ANCHORED WINDOWS. They evaluate at a horizon h BEFORE the failure
     itself, sweeping h from 0 to 56 days. Our landmark grid sits at fixed
     calendar days instead, so a battery failing on day 431 is seen 11 days out
     while one failing on day 530 is seen 110 days out. Anchoring each positive
     to its own failure date is what makes the horizon axis comparable.

  3. AGE-MATCHED CONTROLS. Positives are drawn near end of life, so they are old
     by construction. If controls are sampled at any age, the classifier can
     separate them on age alone and the score is meaningless. Each control is
     matched to a case's age.

The threshold for balanced accuracy is chosen on the TRAINING folds and applied
to the test fold — picking it on the test set would inflate every number.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
HORIZONS = (0, 7, 14, 28, 42, 56, 84)
WINDOW = 90


def build_case_control(df, feats, horizon: int, window: int, rng):
    """Cases anchored `horizon` days before their failure; age-matched controls."""
    from survival_data import _window_features

    cases, controls = [], []
    surv_pool = []

    for bid, g in df.groupby("ID"):
        g = g.sort_values("day")
        lifetime = float(g.lifetime_days.iloc[0])
        failed = not bool(g.still_alive.iloc[0])
        if failed:
            end = lifetime - horizon
            if end - window < 60:
                continue
            f = _window_features(g, feats, end - window, end)
            if f:
                cases.append((bid, float(end), f))
        else:
            surv_pool.append((bid, g, lifetime))

    if not cases or not surv_pool:
        return None

    # match each control to a case's age, so age cannot separate the classes
    ages = np.array([c[1] for c in cases])
    order = rng.permutation(len(surv_pool))
    for k in order:
        bid, g, lifetime = surv_pool[k]
        target = float(rng.choice(ages))
        end = min(target, lifetime - 1)
        if end - window < 60:
            continue
        f = _window_features(g, feats, end - window, end)
        if f:
            controls.append((bid, float(end), f))
        if len(controls) >= len(cases):
            break

    import pandas as pd
    rows = cases + controls
    X = pd.DataFrame([r[2] for r in rows])
    return (X.to_numpy(np.float64),
            np.r_[np.ones(len(cases)), np.zeros(len(controls))].astype(int),
            np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            list(X.columns))


def balanced_accuracy_cv(x, y, groups, seed=7, folds=5):
    """Grouped 5-fold; threshold picked on train, applied to test."""
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier

    zoo = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler(),
                                  LogisticRegression(max_iter=3000,
                                                     class_weight="balanced")),
        "grad boost": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           max_leaf_nodes=15, min_samples_leaf=15,
                                           l2_regularization=1.0,
                                           random_state=seed)),
    }
    out = {}
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for name, proto in zoo.items():
        bas, aucs = [], []
        for tr, te in cv.split(x, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = clone(proto).fit(x[tr], y[tr])
            p_tr, p_te = m.predict_proba(x[tr])[:, 1], m.predict_proba(x[te])[:, 1]
            # Youden's J on the TRAINING fold only
            fpr, tpr, thr = roc_curve(y[tr], p_tr)
            t = thr[int(np.argmax(tpr - fpr))]
            bas.append(balanced_accuracy_score(y[te], (p_te >= t).astype(int)))
            aucs.append(roc_auc_score(y[te], p_te))
        if bas:
            out[name] = {"bal_acc": float(np.mean(bas)),
                         "bal_acc_sd": float(np.std(bas)),
                         "auc": float(np.mean(aucs))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "aitio_comparison.json")
    args = ap.parse_args()

    import pandas as pd
    from survival_data import DROP

    df = pd.read_parquet(args.daily)
    feats = [c for c in df.columns if c not in DROP]
    rng = np.random.default_rng(args.seed)

    print("\nReplicating Aitio & Howey (Joule 2021) protocol")
    print("  balanced accuracy | failure-anchored windows | age-matched controls")
    print("  grouped 5-fold CV | threshold from Youden's J on the TRAIN fold\n")

    published = {0: 82.0, 56: 73.0}
    bench = {0: 66.0, 56: 51.0}

    hdr = (f"{'horizon':>9s} {'cases':>6s} {'ctrls':>6s} "
           f"{'logistic BA':>13s} {'GB BA':>9s} {'best AUC':>9s} "
           f"{'published':>10s} {'their bench':>12s}")
    print(hdr); print("-" * len(hdr))

    results = {}
    for h in HORIZONS:
        built = build_case_control(df, feats, h, WINDOW, rng)
        if built is None:
            continue
        x, y, grp, age, cols = built
        res = balanced_accuracy_cv(x, y, grp, seed=args.seed)
        if not res:
            continue
        results[h] = {"n_cases": int(y.sum()), "n_controls": int((1 - y).sum()),
                      "models": res,
                      "mean_age_cases": float(age[y == 1].mean()),
                      "mean_age_controls": float(age[y == 0].mean())}
        best_auc = max(r["auc"] for r in res.values())
        pub = f"{published[h]:.0f}%" if h in published else "—"
        bch = f"{bench[h]:.0f}%" if h in bench else "—"
        print(f"{h:7d} d {int(y.sum()):6d} {int((1-y).sum()):6d} "
              f"{res['logistic']['bal_acc']*100:11.1f}% "
              f"{res['grad boost']['bal_acc']*100:8.1f}% "
              f"{best_auc:9.3f} {pub:>10s} {bch:>12s}")

    print("\n" + "=" * 78)
    print("HEAD TO HEAD, on their metric")
    print("=" * 78)
    for h in (0, 56):
        if h not in results:
            continue
        ours = max(r["bal_acc"] for r in results[h]["models"].values()) * 100
        print(f"  {h:2d} days ahead   ours {ours:5.1f}%   "
              f"published {published[h]:5.1f}%   their benchmark {bench[h]:5.1f}%   "
              f"→ {'we lead' if ours > published[h] else 'they lead'} "
              f"by {abs(ours-published[h]):.1f} pts")

    print("\nAGE-MATCHING CHECK (cases vs controls should be close):")
    for h in sorted(results):
        r = results[h]
        print(f"  {h:2d} d: cases {r['mean_age_cases']:.0f} d, "
              f"controls {r['mean_age_controls']:.0f} d, "
              f"gap {abs(r['mean_age_cases']-r['mean_age_controls']):.0f} d")

    print("\nCAVEATS")
    print("  * They used a Gaussian Process classifier on SoH + its gradient +")
    print("    stress factors, computed from the full 60-second sequences.")
    print("    We use daily aggregates over a 90-day window, no SoH estimate.")
    print("  * Their controls were sampled from field workshop repair records;")
    print("    ours are age-matched survivors. Similar intent, not identical.")
    print("  * Same dataset, same 1,027 batteries — so this is closer than any")
    print("    cross-dataset comparison, but it is still a replication, not a")
    print("    re-run of their code.")

    args.out.write_text(json.dumps(
        {"results": {str(k): v for k, v in results.items()},
         "published": published, "their_benchmark": bench,
         "window_days": WINDOW}, indent=2, default=float))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
