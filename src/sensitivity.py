#!/usr/bin/env python
"""
Honesty check on the headline AUC.

The shortest life in this fleet is 401 days. So at landmark day 180, failure
within 28 days is ARITHMETICALLY IMPOSSIBLE — every such row is a guaranteed
negative, and "is this battery old enough to be at risk?" separates them
perfectly. Including those landmarks inflates the AUC without the model having
learned anything about battery health.

This script re-runs the two best models on progressively harder subsets:

  all landmarks   the headline number, easy negatives included
  >= 360 d        only landmarks where a 28-day failure is possible
  >= 420 d        the genuinely at-risk population
  age removed     what the TELEMETRY alone contributes, with age withheld

The last row is the one that says whether the sensors are earning their place.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def score(d, idx, cols, horizons, seed, folds=5):
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_ipcw

    x, dur, ev, grp = d.x[np.ix_(idx, cols)], d.duration[idx], d.event[idx], d.groups[idx]
    y = np.array(list(zip(ev, dur)), dtype=[("event", bool), ("time", float)])
    if ev.sum() < 40 or len(np.unique(grp)) < folds * 3:
        return None

    model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1.0))
    aucs = {h: [] for h in horizons}
    unos = []
    for tr, te in GroupKFold(n_splits=folds).split(x, ev, grp):
        try:
            m = clone(model).fit(x[tr], y[tr])
            risk = m.predict(x[te])
        except Exception:
            continue
        tau = min(dur[tr][ev[tr]].max(), dur[te][ev[te]].max()) * 0.9
        try:
            unos.append(concordance_index_ipcw(y[tr], y[te], risk, tau=tau)[0])
        except Exception:
            pass
        for h in horizons:
            lab = ev[te] & (dur[te] <= h)
            known = ev[te] | (dur[te] > h)
            if lab.sum() >= 5 and (~lab[known]).sum() >= 5:
                aucs[h].append(roc_auc_score(lab[known], risk[known]))
    return {"n": int(len(idx)), "events": int(ev.sum()),
            "uno_c": float(np.mean(unos)) if unos else np.nan,
            **{f"auc_{h}": (float(np.mean(aucs[h])) if aucs[h] else np.nan)
               for h in horizons}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--horizons", type=int, nargs="+", default=[28, 84])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from survival_data import build, verify

    d = build(args.daily)
    verify(d)
    all_cols = list(range(d.x.shape[1]))
    age_i = d.columns.index("age_days")
    no_age = [i for i in all_cols if i != age_i]
    age_only = [age_i]

    H = args.horizons
    print("\nCox PH (all features), grouped 5-fold CV\n")
    hdr = (f"{'subset':34s} {'rows':>6s} {'events':>7s} {'Uno C':>7s}"
           + "".join(f" {'AUC@'+str(h):>8s}" for h in H))
    print(hdr); print("-" * len(hdr))

    rows = []
    for label, sel, cols in (
            ("all landmarks (headline)", np.ones(len(d), bool), all_cols),
            ("landmark >= 360 d", d.landmark >= 360, all_cols),
            ("landmark >= 420 d", d.landmark >= 420, all_cols),
            ("landmark >= 480 d", d.landmark >= 480, all_cols),
            ("AGE ONLY, all landmarks", np.ones(len(d), bool), age_only),
            ("AGE ONLY, landmark >= 420 d", d.landmark >= 420, age_only),
            ("NO AGE, all landmarks", np.ones(len(d), bool), no_age),
            ("NO AGE, landmark >= 420 d", d.landmark >= 420, no_age),
    ):
        r = score(d, np.flatnonzero(sel), cols, H, args.seed)
        if r is None:
            print(f"{label:34s} {'— too few events':>30s}")
            continue
        rows.append((label, r))
        print(f"{label:34s} {r['n']:6d} {r['events']:7d} {r['uno_c']:7.3f}"
              + "".join(f" {r[f'auc_{h}']:8.3f}" for h in H))

    print("\nREADING THIS TABLE")
    print("  Compare 'landmark >= 420' against 'AGE ONLY, landmark >= 420'.")
    print("  That difference is what the telemetry contributes once the easy")
    print("  young-vs-old separation is taken away. If it is small, the sensors")
    print("  are not carrying the prediction — age is.")


if __name__ == "__main__":
    main()
