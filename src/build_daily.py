#!/usr/bin/env python
"""
Stage 1 — raw BBOXX telemetry → per-battery-per-day feature table.

    python build_daily.py --root "/home/dikshant/lead acid" --out daily.parquet

Standalone: imports nothing from mambaRUL_studio. The OCV-SOC curve is derived
here, from the GITT lab file that ships with the dataset, so this stage depends
only on the dataset itself.

A DAY is the natural unit for this data. The telemetry is 60-second, but a solar
home system's cycle is diurnal — one charge from the panel, one discharge
overnight — so a day is one cycle, and daily aggregates are the analogue of the
per-cycle features a lithium model consumes.

Conventions fixed here, once:
  * source current is NEGATIVE for charging; flipped so positive = charge
  * voltage collapses to ~0.01 V during disconnects; those samples are dropped
  * sampling is non-uniform, so every integral clamps its timestep
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = 6
NOMINAL_AH = 20.0
MAX_GAP_S = 900.0
MIN_SAMPLES_PER_DAY = 200          # a day with less than this is not usable


# ── OCV-SOC curve from the GITT lab file ─────────────────────────────────────

def ocv_curve_from_gitt(path: Path, n_grid: int = 21):
    """Relaxed cell voltage at the end of each GITT rest → OCV vs SOC.

    Charge and discharge branches are averaged: lead-acid shows real OCV
    hysteresis and using one branch would bias every derived SOC one way.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    q_col = next(c for c in df.columns if c.startswith("_Q-Qo"))

    t = df["time_s"].to_numpy(float)
    v = df["Ewe_V"].to_numpy(float) / CELLS
    i = df["I_mA"].to_numpy(float)
    q = df[q_col].to_numpy(float)
    hc = df["half cycle"].to_numpy(float)

    rest = np.abs(i) < 1.0
    flip = np.diff(rest.astype(np.int8))
    starts = list(np.where(flip == 1)[0] + 1)
    ends = list(np.where(flip == -1)[0])
    if rest[0]:
        starts.insert(0, 0)
    if rest[-1]:
        ends.append(len(rest) - 1)

    pts = [(q[e], v[e], hc[e]) for s, e in zip(starts, ends)
           if e > s and (t[e] - t[s]) > 600.0]
    if len(pts) < 5:
        raise ValueError(f"only {len(pts)} GITT rest points")

    qq = np.array([p[0] for p in pts])
    vv = np.array([p[1] for p in pts])
    bb = np.array([p[2] for p in pts])
    soc = (qq - qq.min()) / (qq.max() - qq.min()) * 100.0

    grid = np.linspace(0.0, 100.0, n_grid)
    branches = []
    for b in np.unique(bb):
        m = bb == b
        if m.sum() >= 4:
            o = np.argsort(soc[m])
            branches.append(np.interp(grid, soc[m][o], vv[m][o]))
    ocv = np.maximum.accumulate(np.mean(branches, axis=0))
    return grid, ocv


def soc_from_voltage(v_pack, grid, ocv):
    """Pack volts → SOC %. Only meaningful on rested samples."""
    return np.interp(v_pack / CELLS, ocv, grid)


# ── per-battery daily aggregation ────────────────────────────────────────────

def daily_features(t, i, v, temp, grid, ocv) -> pd.DataFrame:
    """One row per calendar day of operation."""
    day = ((t - t[0]) // 86400).astype(np.int64)
    dt = np.clip(np.diff(t, prepend=t[0]), 0.0, MAX_GAP_S)

    soc = soc_from_voltage(v, grid, ocv)
    rest = np.abs(i) < 0.05
    chg = i > 0.05
    dis = i < -0.05

    ah = i * dt / 3600.0
    ah_in = np.where(chg, ah, 0.0)
    ah_out = np.where(dis, -ah, 0.0)

    order = np.argsort(day, kind="stable")
    day_s = day[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(day_s)) + 1, day_s.size]
    days = day_s[bounds[:-1]]

    def agg(arr, fn):
        a = arr[order]
        return np.array([fn(a[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])])

    n = agg(np.ones_like(t), np.sum)
    rows = pd.DataFrame({
        "day": days,
        "n_samples": n.astype(int),
        "v_mean": agg(v, np.mean),
        "v_min": agg(v, np.min),
        "v_max": agg(v, np.max),
        "v_std": agg(v, np.std),
        "t_mean": agg(temp, np.mean),
        "t_max": agg(temp, np.max),
        "t_min": agg(temp, np.min),
        "ah_in": agg(ah_in, np.sum),
        "ah_out": agg(ah_out, np.sum),
        "soc_mean": agg(soc, np.mean),
        "soc_min": agg(soc, np.min),
        "soc_max": agg(soc, np.max),
        "frac_rest": agg(rest.astype(float), np.mean),
        "frac_chg": agg(chg.astype(float), np.mean),
        "i_max_dis": agg(np.where(dis, -i, 0.0), np.max),
        "i_mean_chg": agg(np.where(chg, i, np.nan),
                          lambda a: np.nanmean(a) if np.isfinite(a).any() else 0.0),
        "v_rest_med": agg(np.where(rest, v, np.nan),
                          lambda a: np.nanmedian(a) if np.isfinite(a).any() else np.nan),
        "soc_rest_mean": agg(np.where(rest, soc, np.nan),
                             lambda a: np.nanmean(a) if np.isfinite(a).any() else np.nan),
        "v_top_chg": agg(np.where(chg, v, np.nan),
                         lambda a: np.nanmax(a) if np.isfinite(a).any() else np.nan),
    })
    rows = rows[rows.n_samples >= MIN_SAMPLES_PER_DAY].reset_index(drop=True)
    if rows.empty:
        return rows

    rows["soc_swing"] = rows.soc_max - rows.soc_min
    rows["eq_cycles"] = rows.ah_out / NOMINAL_AH
    rows["coul_eff"] = np.where(rows.ah_in > 0.05, rows.ah_out / rows.ah_in, np.nan)
    rows["v_range"] = rows.v_max - rows.v_min
    rows["t_range"] = rows.t_max - rows.t_min
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="BBOXX telemetry → daily features")
    ap.add_argument("--root", type=Path, default=Path.home() / "lead acid")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "daily.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = all batteries")
    args = ap.parse_args()

    grid, ocv = ocv_curve_from_gitt(args.root / "GITT_OCV.mpt")
    print(f"OCV curve: {ocv[0]:.4f} → {ocv[-1]:.4f} V/cell over {len(grid)} points")

    meta = pd.read_csv(args.root / "meta_data.csv").set_index("ID")

    index: dict[int, tuple[str, str]] = {}
    for zf in sorted(args.root.glob("set_*.zip")):
        with zipfile.ZipFile(zf) as z:
            for name in z.namelist():
                if Path(name).stem.isdigit():
                    index[int(Path(name).stem)] = (zf.name, name)
    ids = sorted(index)
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} batteries")

    handles: dict[str, zipfile.ZipFile] = {}
    frames, t0 = [], time.time()
    for k, bid in enumerate(ids, 1):
        zf, member = index[bid]
        if zf not in handles:
            handles[zf] = zipfile.ZipFile(args.root / zf)
        try:
            with handles[zf].open(member) as f:
                raw = np.load(io.BytesIO(f.read()))["arr_0"]
            t, i, v, temp = raw[:, 0], -raw[:, 1], raw[:, 2], raw[:, 3]
            ok = (np.isfinite(t) & np.isfinite(i) & np.isfinite(v)
                  & np.isfinite(temp) & (v > 6.0) & (v < 16.0)
                  & (temp > -20.0) & (temp < 80.0))
            t, i, v, temp = t[ok], i[ok], v[ok], temp[ok]
            o = np.argsort(t)
            t, i, v, temp = t[o], i[o], v[o], temp[o]
            if t.size < 20000:
                continue
            d = daily_features(t, i, v, temp, grid, ocv)
            if d.empty:
                continue
            d.insert(0, "ID", bid)
            d["lifetime_days"] = int(meta.loc[bid, "Lifetime"])
            d["still_alive"] = bool(meta.loc[bid, "STILL_ALIVE"])
            frames.append(d)
        except Exception as exc:
            print(f"  battery {bid}: {type(exc).__name__}: {exc}")
        if k % 25 == 0 or k == len(ids):
            r = k / max(time.time() - t0, 1e-9)
            print(f"\r  {k}/{len(ids)}  {r:.1f}/s  eta {(len(ids)-k)/max(r,1e-9):.0f}s",
                  end="", flush=True)
    for h in handles.values():
        h.close()

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    (args.out.parent / "ocv_curve.json").write_text(json.dumps(
        {"soc_pct": grid.tolist(), "ocv_v_per_cell": ocv.tolist()}, indent=2))

    print(f"\n\nwrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    print(f"  {len(out)} battery-days across {out.ID.nunique()} batteries")
    print(f"  days per battery: median {out.groupby('ID').size().median():.0f}")
    print(f"  failed {out[~out.still_alive].ID.nunique()} | "
          f"censored {out[out.still_alive].ID.nunique()}")


if __name__ == "__main__":
    main()
