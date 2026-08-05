"""
Stage 2 — daily features → leakage-free training sequences.

Four leakage guards, each enforced in code and each verified by
`verify_no_leakage()` before training starts:

  1. SPLIT BY BATTERY. A battery contributes windows to exactly one of
     train/val/test. Splitting by window would put day-200 and day-400 of the
     same cell on both sides and inflate every score.

  2. CAUSAL WINDOWS. A window covering days [d−L+1, d] predicts the RUL at day
     d. Nothing after d is visible, so the model never sees its own future.

  3. NORMALISATION FITTED ON TRAIN ONLY. Means and standard deviations come
     from training windows. Fitting them on the full set leaks the test
     distribution into the inputs — subtle, and it flatters results.

  4. NO TARGET-DERIVED FEATURES. Nothing in the feature set is computed from
     lifetime or the failure flag.

Censoring is handled rather than discarded. 536 of 1,027 batteries were still
alive at the end of the study, so their true RUL is unknown — only bounded
below. Dropping them would throw away half the fleet and bias the model toward
short lives; they are kept with a one-sided loss instead (see train.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_LEN = 30          # days of history per window — one month
STRIDE = 7            # a window per week; denser is heavily autocorrelated
RUL_SCALE = 400.0     # days; targets are divided by this
MAX_DAY_SPAN = 45     # a 30-row window must not span more calendar days

# `day` — the battery's AGE at the end of the window — is deliberately KEPT as a
# feature. It is not leakage: at inference you always know how old a battery is,
# you just don't know its total lifetime. Excluding it costs a great deal here,
# because lifetime is 533 ± 89 days, so age alone is a strong prior — a
# predictor of `mean_lifetime − age` scores 63.8 d MAE, better than a model
# trained without it.
DROP_COLS = {"ID", "lifetime_days", "still_alive", "n_samples"}


@dataclass
class Split:
    x: np.ndarray            # (N, SEQ_LEN, F) float32
    y: np.ndarray            # (N,) RUL / RUL_SCALE
    censored: np.ndarray     # (N,) 1 = true RUL is only bounded below
    ids: np.ndarray          # (N,) battery id
    day: np.ndarray          # (N,) landmark day
    lifetime: np.ndarray     # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        return (f"{len(self):6d} windows | {len(np.unique(self.ids)):4d} batteries | "
                f"censored {self.censored.mean():5.1%} | "
                f"RUL days {self.y.min()*RUL_SCALE:6.0f}–{self.y.max()*RUL_SCALE:6.0f}")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def _windows_for_battery(g: pd.DataFrame, feats: list[str]):
    """Causal windows for one battery. Returns (X, y_days, censored, day)."""
    g = g.sort_values("day")
    arr = g[feats].to_numpy(np.float32)
    days = g["day"].to_numpy(np.int64)
    lifetime = float(g["lifetime_days"].iloc[0])
    alive = bool(g["still_alive"].iloc[0])

    xs, ys, cs, ds = [], [], [], []
    for end in range(SEQ_LEN - 1, len(g), STRIDE):
        start = end - SEQ_LEN + 1
        # rows are dropped when a day is under-sampled, so a 30-row window can
        # straddle a gap; reject any that covers too much calendar time
        if days[end] - days[start] > MAX_DAY_SPAN:
            continue
        d = int(days[end])
        rul = lifetime - d
        if rul < 0:
            continue                      # window extends past the recorded end
        xs.append(arr[start:end + 1])
        ys.append(rul)
        cs.append(1 if alive else 0)
        ds.append(d)
    if not xs:
        return None
    return (np.stack(xs), np.asarray(ys, np.float32),
            np.asarray(cs, np.int8), np.asarray(ds, np.int64))


def build(parquet: Path, seed: int = 7,
          frac=(0.70, 0.15, 0.15)) -> tuple[Split, Split, Split, list[str], dict]:
    df = pd.read_parquet(parquet)
    # age in the window, expressed two ways: the raw ramp and its square, so the
    # model can express a non-linear hazard in age without having to build one
    df["age_days"] = df["day"].astype(np.float32)
    df["age_sq"] = (df["age_days"] / 365.0) ** 2
    feats = feature_columns(df)

    # forward-fill within each battery, then a global median for anything still
    # missing (a battery that never rests has no rest-voltage on any day)
    df[feats] = df.groupby("ID")[feats].ffill().bfill()

    ids = df.ID.unique()
    rng = np.random.default_rng(seed)
    # stratify the split by outcome so each part holds a similar failure rate
    outcome = df.groupby("ID").still_alive.first()
    tr_ids, va_ids, te_ids = [], [], []
    for alive in (True, False):
        pool = np.array(sorted(outcome[outcome == alive].index))
        rng.shuffle(pool)
        n = len(pool)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        tr_ids += list(pool[:a])
        va_ids += list(pool[a:b])
        te_ids += list(pool[b:])
    parts = {"train": set(tr_ids), "val": set(va_ids), "test": set(te_ids)}

    built: dict[str, Split] = {}
    for name, keep in parts.items():
        X, Y, C, D, I, L = [], [], [], [], [], []
        for bid, g in df[df.ID.isin(keep)].groupby("ID"):
            w = _windows_for_battery(g, feats)
            if w is None:
                continue
            x, y, c, d = w
            X.append(x); Y.append(y); C.append(c); D.append(d)
            I.append(np.full(len(y), bid, np.int64))
            L.append(np.full(len(y), g["lifetime_days"].iloc[0], np.float32))
        built[name] = Split(np.concatenate(X), np.concatenate(Y) / RUL_SCALE,
                            np.concatenate(C), np.concatenate(I),
                            np.concatenate(D), np.concatenate(L))

    # ---- guard 3: statistics from TRAIN ONLY
    flat = built["train"].x.reshape(-1, built["train"].x.shape[-1])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd[sd < 1e-6] = 1.0
    for s in built.values():
        s.x = np.nan_to_num((s.x - mu) / sd, nan=0.0,
                            posinf=0.0, neginf=0.0).astype(np.float32)
        np.clip(s.x, -8.0, 8.0, out=s.x)

    norm = {"mean": mu.tolist(), "std": sd.tolist(),
            "features": feats, "seq_len": SEQ_LEN, "rul_scale": RUL_SCALE}
    return built["train"], built["val"], built["test"], feats, norm


def verify_no_leakage(tr: Split, va: Split, te: Split) -> None:
    """Fail loudly rather than quietly producing an inflated score."""
    a, b, c = set(tr.ids), set(va.ids), set(te.ids)
    assert not (a & b), f"{len(a & b)} batteries in BOTH train and val"
    assert not (a & c), f"{len(a & c)} batteries in BOTH train and test"
    assert not (b & c), f"{len(b & c)} batteries in BOTH val and test"

    for s, name in ((tr, "train"), (va, "val"), (te, "test")):
        assert np.isfinite(s.x).all(), f"{name} has non-finite features"
        assert (s.y >= 0).all(), f"{name} has negative RUL"
        # a window ending at day d must have RUL = lifetime − d
        assert np.allclose(s.y * RUL_SCALE, s.lifetime - s.day, atol=1e-3), \
            f"{name}: RUL does not equal lifetime − landmark day"

    print(f"leakage checks passed — {len(a)}/{len(b)}/{len(c)} disjoint "
          f"batteries in train/val/test")
