#!/usr/bin/env python
"""
Paper-ready results: full model × subset comparison, ablation, calibration,
feature attribution and bootstrap intervals.

    python paper_results.py

Four things this produces that the first comparison did not:

  TABLE 1  every model on both the full landmark set and the AT-RISK subset.
           The full set contains landmarks where failure is arithmetically
           impossible (shortest life is 401 days, so nothing can fail within
           28 days of day 180) and those free negatives inflate AUC. The
           at-risk column is the number that should be quoted.

  TABLE 2  ablation — full features vs age-only vs telemetry-only, per model,
           on the at-risk subset. Answers "do the sensors earn their place?"
           separately for each algorithm rather than only for Cox.

  TABLE 3  calibration. For a subscription alert product a well-ranked but
           badly-calibrated score is close to useless: the customer needs
           "this is a 40 % risk", not "this is riskier than that one". Brier
           score plus a reliability curve.

  TABLE 4  permutation importance on the best model, so the paper can say what
           physically drives the prediction.

Bootstrap confidence intervals are over BATTERIES, not rows, because rows from
one battery are not independent.
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
H = (28, 84)


def models(seed: int) -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import (ExtraSurvivalTrees,
                                 GradientBoostingSurvivalAnalysis,
                                 RandomSurvivalForest)
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

    return {
        "Cox PH": make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1.0)),
        "Coxnet": make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True)),
        "RSF": RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                    max_features="sqrt", n_jobs=-1,
                                    random_state=seed),
        "Extra Surv Trees": ExtraSurvivalTrees(
            n_estimators=200, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=seed),
        "Grad Boost Surv": GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed),
    }


def run_cv(proto, x, dur, ev, grp, folds=5, want_prob=True):
    """Grouped CV. Returns out-of-fold risk scores and survival probabilities."""
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    y = np.array(list(zip(ev, dur)), dtype=[("event", bool), ("time", float)])
    risk = np.full(len(dur), np.nan)
    prob = {h: np.full(len(dur), np.nan) for h in H}
    for tr, te in GroupKFold(n_splits=folds).split(x, ev, grp):
        try:
            m = clone(proto).fit(x[tr], y[tr])
            risk[te] = m.predict(x[te])
            if want_prob:
                fns = m.predict_survival_function(x[te], return_array=False)
                for h in H:
                    prob[h][te] = 1.0 - np.array([f(h) for f in fns])
        except Exception:
            continue
    return risk, prob, y


def metrics(risk, prob, dur, ev, y):
    from sklearn.metrics import roc_auc_score
    from sksurv.metrics import concordance_index_ipcw

    ok = np.isfinite(risk)
    out = {}
    try:
        tau = np.percentile(dur[ev], 90)
        out["uno_c"] = float(concordance_index_ipcw(y[ok], y[ok], risk[ok], tau=tau)[0])
    except Exception:
        out["uno_c"] = np.nan
    for h in H:
        lab = ev & (dur <= h)
        known = (ev | (dur > h)) & ok
        out[f"auc_{h}"] = (float(roc_auc_score(lab[known], risk[known]))
                           if lab[known].sum() >= 5 else np.nan)
        p = prob[h]
        m = known & np.isfinite(p)
        out[f"brier_{h}"] = (float(np.mean((p[m] - lab[m]) ** 2))
                             if m.sum() > 20 else np.nan)
    return out


def boot_ci(risk, dur, ev, grp, h, n=400, seed=7):
    """Bootstrap over batteries — rows within a battery are not independent."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    bats = np.unique(grp)
    lab = ev & (dur <= h)
    known = ev | (dur > h)
    vals = []
    for _ in range(n):
        pick = rng.choice(bats, len(bats), replace=True)
        idx = np.concatenate([np.flatnonzero(grp == b) for b in pick])
        idx = idx[known[idx] & np.isfinite(risk[idx])]
        if lab[idx].sum() >= 5 and (~lab[idx]).sum() >= 5:
            vals.append(roc_auc_score(lab[idx], risk[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
        if len(vals) > 50 else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--at-risk-landmark", type=int, default=420)
    ap.add_argument("--out", type=Path, default=HERE / "paper_results.json")
    args = ap.parse_args()

    from survival_data import build, verify

    d = build(args.daily)
    verify(d)
    age_i = d.columns.index("age_days")
    all_c = list(range(d.x.shape[1]))
    no_age = [i for i in all_c if i != age_i]
    at_risk = d.landmark >= args.at_risk_landmark

    print(f"\nat-risk subset = landmark >= {args.at_risk_landmark} d : "
          f"{at_risk.sum()} rows, {d.event[at_risk].sum()} events, "
          f"{len(np.unique(d.groups[at_risk]))} batteries\n")

    store: dict = {}

    # ── TABLE 1 ───────────────────────────────────────────────────────────
    print("=" * 96)
    print("TABLE 1 — all models, full landmark set vs AT-RISK subset")
    print("=" * 96)
    print(f"{'model':18s} | {'FULL SET (inflated)':^33s} | {'AT-RISK (quote this)':^33s}")
    print(f"{'':18s} | {'UnoC':>7s} {'AUC@4w':>8s} {'AUC@12w':>8s} {'Br@4w':>7s} |"
          f" {'UnoC':>7s} {'AUC@4w':>8s} {'AUC@12w':>8s} {'Br@4w':>7s}")
    print("-" * 96)
    for name, proto in models(args.seed).items():
        row = {}
        for tag, sel in (("full", np.ones(len(d), bool)), ("at_risk", at_risk)):
            i = np.flatnonzero(sel)
            r, p, y = run_cv(proto, d.x[np.ix_(i, all_c)], d.duration[i],
                             d.event[i], d.groups[i])
            row[tag] = metrics(r, {h: p[h] for h in H}, d.duration[i], d.event[i], y)
            if tag == "at_risk":
                row["risk_at_risk"] = r
        store[name] = {k: v for k, v in row.items() if k != "risk_at_risk"}
        f, a = row["full"], row["at_risk"]
        print(f"{name:18s} | {f['uno_c']:7.3f} {f['auc_28']:8.3f} {f['auc_84']:8.3f} "
              f"{f['brier_28']:7.4f} | {a['uno_c']:7.3f} {a['auc_28']:8.3f} "
              f"{a['auc_84']:8.3f} {a['brier_28']:7.4f}")
        store[name]["_risk"] = row["risk_at_risk"]

    # ── TABLE 2 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"TABLE 2 — ablation on the AT-RISK subset: do the sensors earn their place?")
    print("=" * 78)
    print(f"{'model':18s} {'features':16s} {'UnoC':>7s} {'AUC@4w':>8s} {'AUC@12w':>8s}")
    print("-" * 78)
    i = np.flatnonzero(at_risk)
    abl: dict = {}
    for name, proto in models(args.seed).items():
        abl[name] = {}
        for tag, cols in (("all", all_c), ("age only", [age_i]),
                          ("telemetry only", no_age)):
            r, p, y = run_cv(proto, d.x[np.ix_(i, cols)], d.duration[i],
                             d.event[i], d.groups[i], want_prob=False)
            m = metrics(r, {h: np.full(len(i), np.nan) for h in H},
                        d.duration[i], d.event[i], y)
            abl[name][tag] = m
            print(f"{name if tag=='all' else '':18s} {tag:16s} {m['uno_c']:7.3f} "
                  f"{m['auc_28']:8.3f} {m['auc_84']:8.3f}")
        print("-" * 78)

    # ── TABLE 3 ───────────────────────────────────────────────────────────
    best_brier = min(store, key=lambda k: store[k]["at_risk"]["brier_28"])
    best_auc = max(store, key=lambda k: store[k]["at_risk"]["auc_28"])
    print("\n" + "=" * 78)
    print(f"TABLE 3 — calibration of the best-Brier model ({best_brier}), AT-RISK, 4 weeks")
    print("=" * 78)
    proto = models(args.seed)[best_brier]
    r, p, y = run_cv(proto, d.x[np.ix_(i, all_c)], d.duration[i], d.event[i],
                     d.groups[i])
    lab = d.event[i] & (d.duration[i] <= 28)
    known = (d.event[i] | (d.duration[i] > 28)) & np.isfinite(p[28])
    pr, lb = p[28][known], lab[known]
    print(f"{'predicted risk':>18s} {'n':>6s} {'predicted':>10s} {'observed':>10s}")
    edges = np.quantile(pr, np.linspace(0, 1, 6))
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pr >= lo) & (pr <= hi)
        if m.sum() > 10:
            calib.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                          "pred": float(pr[m].mean()), "obs": float(lb[m].mean())})
            print(f"{lo:8.3f}–{hi:<8.3f} {m.sum():6d} {pr[m].mean():10.3f} "
                  f"{lb[m].mean():10.3f}")

    # ── TABLE 4 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"TABLE 4 — permutation importance, {best_auc}, AT-RISK, 4-week AUC")
    print("=" * 78)
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    x_ar, dur_ar, ev_ar, grp_ar = (d.x[np.ix_(i, all_c)], d.duration[i],
                                   d.event[i], d.groups[i])
    y_ar = np.array(list(zip(ev_ar, dur_ar)), dtype=[("event", bool), ("time", float)])
    tr, te = next(iter(GroupKFold(n_splits=5).split(x_ar, ev_ar, grp_ar)))
    fitted = clone(models(args.seed)[best_auc]).fit(x_ar[tr], y_ar[tr])
    lab_te = ev_ar[te] & (dur_ar[te] <= 28)
    kn = ev_ar[te] | (dur_ar[te] > 28)
    base = roc_auc_score(lab_te[kn], fitted.predict(x_ar[te])[kn])
    rng = np.random.default_rng(args.seed)
    imps = []
    for j in range(x_ar.shape[1]):
        drops = []
        for _ in range(5):
            xp = x_ar[te].copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops.append(base - roc_auc_score(lab_te[kn], fitted.predict(xp)[kn]))
        imps.append(float(np.mean(drops)))
    order = np.argsort(imps)[::-1][:15]
    for k in order:
        print(f"   {d.columns[k]:30s} {imps[k]:+.4f}  {'#'*max(0,int(imps[k]*400))}")

    # ── headline with bootstrap CI ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("HEADLINE (at-risk population, bootstrapped over batteries)")
    print("=" * 78)
    rk = store[best_auc]["_risk"]
    for h in H:
        lo, hi = boot_ci(rk, dur_ar, ev_ar, grp_ar, h, seed=args.seed)
        print(f"  {best_auc}  AUC@{h//7}wk = {store[best_auc]['at_risk'][f'auc_{h}']:.3f} "
              f"  95% CI [{lo:.3f}, {hi:.3f}]")
    # REMOVED: a hardcoded "Voronov & Frisk ... AUC 0.69 - 0.772" line used to print
    # here. That range appears in NONE of the three Linkoping papers, all read in
    # full: Voronov et al. (IEEE T-Rel 2018) report error rate and concordance and
    # state that Cox "is not applicable" to their data, Frisk et al. (PHM 2014) use
    # AUC only as a variable-importance score, and Frisk & Krysander (IFAC 2015)
    # report neither. It was never sourced, and printing it here would have shipped
    # an unverifiable number alongside the released code.

    for k in store:
        store[k].pop("_risk", None)
    args.out.write_text(json.dumps(
        {"table1": store, "table2_ablation": abl, "table3_calibration": calib,
         "best_auc_model": best_auc, "best_brier_model": best_brier,
         "at_risk_landmark": args.at_risk_landmark}, indent=2, default=float))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
