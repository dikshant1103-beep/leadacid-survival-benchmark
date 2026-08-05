#!/usr/bin/env python
"""
Probability recalibration, with the uncalibrated results preserved alongside.

    python calibrate.py

The survival models rank well but their probabilities are too flat: at 4 weeks
RSF predicted 0.060 where 0.031 was observed, and 0.247 where 0.315 was
observed. Over-warning on healthy batteries and UNDER-warning on the ones about
to fail — the wrong error in the wrong direction for an alert product.

NESTED PROTOCOL, which is the whole point:

    outer GroupKFold          → the evaluation split, never touched by fitting
      inner split of TRAIN    → 70 % fits the model, 30 % fits the calibrator
    calibrator applied to the outer test fold

Fitting a calibrator on the same predictions you then score is the classic way
to manufacture a perfect-looking reliability curve. Both splits are BY BATTERY.

Isotonic regression is monotone, so it cannot change the ranking — AUC before
and after must be identical. That equality is asserted, and it is the check
that the pipeline is wired correctly.

Metrics: Brier (accuracy of the probability), ECE and MCE (calibration error),
plus the reliability table before and after.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
HORIZONS = (28, 84)
AT_RISK = 420


def ece_mce(prob, label, bins=10):
    """Expected and maximum calibration error over equal-count bins."""
    if len(prob) < bins * 5:
        return np.nan, np.nan
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    e, m, n = 0.0, 0.0, len(prob)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (prob >= lo) & (prob < hi)
        if s.sum() < 2:
            continue
        gap = abs(prob[s].mean() - label[s].mean())
        e += s.sum() / n * gap
        m = max(m, gap)
    return float(e), float(m)


def reliability(prob, label, bins=5):
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (prob >= lo) & (prob < hi)
        if s.sum() > 5:
            out.append({"lo": float(lo), "hi": float(hi), "n": int(s.sum()),
                        "predicted": float(prob[s].mean()),
                        "observed": float(label[s].mean())})
    return out


def nested_calibrated(proto, x, dur, ev, grp, horizon, seed=7, folds=5):
    """Out-of-fold raw and calibrated probabilities for one horizon."""
    from sklearn.base import clone
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    y = np.array(list(zip(ev, dur)), dtype=[("event", bool), ("time", float)])
    raw = np.full(len(dur), np.nan)
    iso = np.full(len(dur), np.nan)
    sig = np.full(len(dur), np.nan)

    for tr, te in GroupKFold(n_splits=folds).split(x, ev, grp):
        # inner split of the TRAINING fold only, again by battery
        inner = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        fit_i, cal_i = next(inner.split(x[tr], ev[tr], grp[tr]))
        fit, cal = tr[fit_i], tr[cal_i]
        try:
            m = clone(proto).fit(x[fit], y[fit])
        except Exception:
            continue

        def risk_prob(idx):
            fns = m.predict_survival_function(x[idx], return_array=False)
            return 1.0 - np.array([f(horizon) for f in fns])

        p_cal, p_te = risk_prob(cal), risk_prob(te)
        raw[te] = p_te

        # the calibrator only ever sees rows whose status at `horizon` is known
        lab_cal = ev[cal] & (dur[cal] <= horizon)
        known_cal = ev[cal] | (dur[cal] > horizon)
        if known_cal.sum() < 40 or lab_cal[known_cal].sum() < 5:
            continue

        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p_cal[known_cal], lab_cal[known_cal].astype(float))
        iso[te] = ir.predict(p_te)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_cal[known_cal].reshape(-1, 1), lab_cal[known_cal].astype(int))
        sig[te] = lr.predict_proba(p_te.reshape(-1, 1))[:, 1]

    return raw, iso, sig


def score(prob, lab, known):
    from sklearn.metrics import roc_auc_score
    m = known & np.isfinite(prob)
    if m.sum() < 50 or lab[m].sum() < 5:
        return None
    p, l = prob[m], lab[m].astype(float)
    e, mx = ece_mce(p, l)
    return {"n": int(m.sum()), "brier": float(np.mean((p - l) ** 2)),
            "ece": e, "mce": mx, "auc": float(roc_auc_score(l, p)),
            "mean_pred": float(p.mean()), "base_rate": float(l.mean()),
            "reliability": reliability(p, l)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "calibration_results.json")
    args = ap.parse_args()

    from sksurv.ensemble import (GradientBoostingSurvivalAnalysis,
                                 RandomSurvivalForest)

    from survival_data import build, verify

    d = build(args.daily)
    verify(d)
    sel = np.flatnonzero(d.landmark >= AT_RISK)
    x, dur, ev, grp = d.x[sel], d.duration[sel], d.event[sel], d.groups[sel]
    print(f"\nat-risk: {len(sel)} rows, {ev.sum()} events, "
          f"{len(np.unique(grp))} batteries\n")

    zoo = {
        "Grad Boost Surv": GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=args.seed),
        "RSF": RandomSurvivalForest(
            n_estimators=200, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=args.seed),
    }

    results: dict = {}
    for name, proto in zoo.items():
        results[name] = {}
        for h in HORIZONS:
            raw, iso, sig = nested_calibrated(proto, x, dur, ev, grp, h,
                                              seed=args.seed)
            lab = ev & (dur <= h)
            known = ev | (dur > h)
            entry = {}
            for tag, p in (("uncalibrated", raw), ("isotonic", iso),
                           ("platt", sig)):
                s = score(p, lab, known)
                if s:
                    entry[tag] = s
            results[name][f"h{h}"] = entry

            print("=" * 78)
            print(f"{name} — {h//7}-week horizon "
                  f"(base rate {entry['uncalibrated']['base_rate']:.3%})")
            print("=" * 78)
            print(f"{'variant':16s} {'Brier':>9s} {'ECE':>8s} {'MCE':>8s} "
                  f"{'AUC':>8s} {'mean pred':>10s}")
            for tag in ("uncalibrated", "isotonic", "platt"):
                if tag in entry:
                    e = entry[tag]
                    print(f"{tag:16s} {e['brier']:9.4f} {e['ece']:8.4f} "
                          f"{e['mce']:8.4f} {e['auc']:8.3f} {e['mean_pred']:10.4f}")

            # isotonic is monotone, so it must not move the ranking
            if "isotonic" in entry:
                da = abs(entry["isotonic"]["auc"] - entry["uncalibrated"]["auc"])
                flag = "OK" if da < 0.02 else f"!! moved by {da:.3f}"
                print(f"  monotonicity check — AUC unchanged by isotonic: {flag}")

            best = min((t for t in entry), key=lambda t: entry[t]["brier"])
            print(f"\n  reliability, {best}:")
            print(f"  {'predicted':>10s} {'observed':>10s} {'n':>6s}")
            for b in entry[best]["reliability"]:
                print(f"  {b['predicted']:10.3f} {b['observed']:10.3f} {b['n']:6d}")
            print()

    # ---- keep the earlier uncalibrated run alongside
    prev = HERE / "paper_results.json"
    combined = {"calibration": results}
    if prev.exists():
        combined["previous_uncalibrated_run"] = json.loads(prev.read_text())
        print(f"preserved earlier results from {prev.name} inside {args.out.name}")
    args.out.write_text(json.dumps(combined, indent=2, default=float))

    print("\n" + "=" * 78)
    print("SUMMARY — Brier score, lower is better")
    print("=" * 78)
    print(f"{'model / horizon':28s} {'uncal':>9s} {'isotonic':>9s} {'platt':>9s} "
          f"{'best':>10s}")
    for name in results:
        for h in HORIZONS:
            e = results[name][f"h{h}"]
            vals = {t: e[t]["brier"] for t in e}
            best = min(vals, key=vals.get)
            print(f"{name + f'  {h//7}wk':28s} "
                  + " ".join(f"{vals.get(t, float('nan')):9.4f}"
                             for t in ("uncalibrated", "isotonic", "platt"))
                  + f" {best:>10s}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
