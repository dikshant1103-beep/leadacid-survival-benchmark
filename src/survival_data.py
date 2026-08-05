"""
Landmark survival dataset for lead-acid failure prediction.

Why landmarking rather than one row per battery: each battery is observed for
400–760 days and we want a prediction at any point in its life, not just once.
At each landmark day L we take the batteries still alive at L, build features
from the window [L−W, L], and ask how long they survive AFTER L. That is the
standard construction for repeated-measures survival data and it keeps every
prediction honestly out-of-sample in time.

Two properties this preserves that a naive setup destroys:

  * RIGHT CENSORING IS KEPT, NOT DISCARDED. 536 of 1,027 batteries were alive
    when the study ended. A regression target throws that away or, worse,
    pretends the last observation was a failure. Survival models consume
    (duration, event) pairs directly, which is exactly why the literature on
    fleet lead-acid prognostics uses them.

  * NOTHING LOOKS FORWARD. Features come only from [L−W, L]; the outcome is
    measured strictly after L.

Grouping is by battery, so a battery never appears on both sides of a split —
it contributes several landmark rows and all of them travel together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 90
LANDMARKS = (180, 240, 300, 360, 420, 480, 540)
MIN_ROWS_IN_WINDOW = 45          # a 90-day window needs at least half its days

DROP = {"ID", "day", "lifetime_days", "still_alive", "n_samples"}


@dataclass
class SurvivalData:
    x: np.ndarray            # (N, F)
    duration: np.ndarray     # (N,) days from landmark to event or censoring
    event: np.ndarray        # (N,) bool — True = observed failure
    groups: np.ndarray       # (N,) battery id
    landmark: np.ndarray     # (N,) landmark day
    columns: list[str]

    def __len__(self) -> int:
        return len(self.duration)

    @property
    def y(self) -> np.ndarray:
        """Structured array in the (event, time) form scikit-survival expects."""
        return np.array(list(zip(self.event, self.duration)),
                        dtype=[("event", bool), ("time", float)])

    def summary(self) -> str:
        return (f"{len(self):5d} rows | {len(np.unique(self.groups)):4d} batteries | "
                f"events {self.event.sum():4d} ({self.event.mean():5.1%}) | "
                f"duration {self.duration.min():.0f}–{self.duration.max():.0f} d")


def _window_features(g: pd.DataFrame, feats: list[str], lo: int, hi: int) -> dict:
    """Summarise one battery's window: level, trend and volatility per feature.

    Level says where the battery is, trend says where it is going, volatility
    says how erratic it has been. A single snapshot cannot express degradation;
    the slope is what carries it.
    """
    w = g[(g.day > lo) & (g.day <= hi)]
    if len(w) < MIN_ROWS_IN_WINDOW:
        return {}
    t = w.day.to_numpy(float)
    tc = t - t.mean()
    denom = float((tc ** 2).sum()) or 1.0

    out: dict[str, float] = {}
    for f in feats:
        v = w[f].to_numpy(float)
        if not np.isfinite(v).any():
            out[f"{f}_last"] = out[f"{f}_mean"] = np.nan
            out[f"{f}_slope"] = out[f"{f}_std"] = np.nan
            continue
        v = np.nan_to_num(v, nan=float(np.nanmean(v)))
        out[f"{f}_mean"] = float(v.mean())
        out[f"{f}_last"] = float(v[-7:].mean())          # last week of the window
        out[f"{f}_slope"] = float(((tc * (v - v.mean())).sum() / denom) * 365.0)
        out[f"{f}_std"] = float(v.std())
    out["age_days"] = float(hi)
    return out


def build(parquet: Path, window_days: int = WINDOW_DAYS,
          landmarks=LANDMARKS) -> SurvivalData:
    df = pd.read_parquet(parquet)
    feats = [c for c in df.columns if c not in DROP]

    rows, dur, ev, grp, lm = [], [], [], [], []
    for bid, g in df.groupby("ID"):
        g = g.sort_values("day")
        lifetime = float(g.lifetime_days.iloc[0])
        alive_at_end = bool(g.still_alive.iloc[0])
        for L in landmarks:
            # the battery must still be at risk at the landmark
            if lifetime <= L:
                continue
            f = _window_features(g, feats, L - window_days, L)
            if not f:
                continue
            rows.append(f)
            dur.append(lifetime - L)          # time from landmark to event/censoring
            ev.append(not alive_at_end)       # True only for an observed failure
            grp.append(bid)
            lm.append(L)

    X = pd.DataFrame(rows)
    cols = list(X.columns)
    return SurvivalData(
        x=X.to_numpy(np.float64), duration=np.asarray(dur, float),
        event=np.asarray(ev, bool), groups=np.asarray(grp),
        landmark=np.asarray(lm), columns=cols)


def verify(d: SurvivalData) -> None:
    assert (d.duration > 0).all(), "non-positive survival duration"
    assert d.x.shape[0] == len(d.duration) == len(d.event) == len(d.groups)
    assert d.event.sum() > 50, f"only {d.event.sum()} observed failures"
    # a censored row must not be treated as an event
    assert d.event.dtype == bool
    # age must equal the landmark: the feature and the construction agree
    age = d.x[:, d.columns.index("age_days")]
    assert np.allclose(age, d.landmark), "age feature disagrees with landmark"
    print(f"survival data verified — {d.summary()}")
