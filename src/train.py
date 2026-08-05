#!/usr/bin/env python
"""
Stage 4 — train LeadAcidBiMamba.

    python build_daily.py --root "/home/dikshant/lead acid"
    python train.py --epochs 60

CENSORING. 536 of the 1,027 batteries were still alive when the study ended, so
their remaining life is unknown — bounded below, not observed. Two wrong ways to
handle that: discard them (throws away half the fleet and biases the model
toward short lives) or treat the bound as if it were the truth (teaches the
model that every surviving battery was about to die). Neither is used here.

Observed failures get a heteroscedastic Gaussian negative log-likelihood, so
the model is scored on both its estimate and its confidence. Censored samples
get a ONE-SIDED penalty: predicting more life than the known lower bound costs
nothing, predicting less is punished. That is the only information a censored
label actually carries.

Evaluation is reported on observed failures only, because they are the only
samples with a ground-truth RUL to compare against.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import RUL_SCALE, Split, build, verify_no_leakage
from model import LeadAcidBiMamba, count_parameters

HERE = Path(__file__).resolve().parent


def make_loader(s: Split, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(s.x), torch.from_numpy(s.y),
                      torch.from_numpy(s.censored.astype(np.float32))),
        batch_size=batch, shuffle=shuffle, drop_last=False)


LOG_MAX = float(np.log1p(RUL_SCALE * 2.0))


def to_log_target(y_norm):
    """RUL/RUL_SCALE  →  log1p(days) scaled to ~[0, 1].

    Predicting in log space so a 30-day error at 20 days left costs far more
    than the same error at 500 days. Under a linear target the optimiser is
    indifferent between them and therefore ignores exactly the region that
    matters — v1 was +56 d biased on batteries with under a month to live.
    """
    return torch.log1p(y_norm * RUL_SCALE) / LOG_MAX


def from_log_target(z):
    return torch.expm1((z * LOG_MAX).clamp(max=LOG_MAX)) / RUL_SCALE


def loss_fn(mu, logvar, y_log, censored, censor_weight: float = 1.0,
            logvar_l2: float = 1e-2, use_nll: bool = True):
    """Right-censored log-normal likelihood (an AFT model).

    OBSERVED failure at t:   −log φ((log t − μ)/σ) / σ
    CENSORED, alive past t:  −log P(T > t) = −log Φ((μ − log t)/σ)

    The v1 hinge was wrong in a way that biased every prediction upward: it
    penalised under-prediction on 54 % of the training rows and applied no
    counter-pressure at all, so the optimiser could always reduce loss by
    predicting more life. The survival term above is the correct likelihood —
    a censored battery says "at least this long", and Φ expresses exactly that,
    with a real gradient in both directions once σ is involved.
    """
    obs = 1.0 - censored
    sigma = torch.exp(0.5 * logvar).clamp(min=1e-3)
    z = (y_log - mu) / sigma

    if use_nll:
        observed = 0.5 * z ** 2 + 0.5 * logvar
    else:
        observed = (y_log - mu) ** 2          # warm-up: mean only
    observed_loss = (observed * obs).sum() / obs.sum().clamp(min=1.0)

    # log Φ((μ − log t)/σ), numerically stable
    censored_ll = torch.special.log_ndtr(-z)
    censored_loss = (-censored_ll * censored).sum() / censored.sum().clamp(min=1.0)

    # keeps the variance head from shrinking to win the NLL — v1's PICP fell
    # from 0.98 to 0.55 doing exactly that
    reg = logvar_l2 * (logvar ** 2).mean()
    return observed_loss + censor_weight * censored_loss + reg


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    mus, ys, cs, lvs = [], [], [], []
    for x, y, c in loader:
        m, lv = model(x.to(device))
        mus.append(m.cpu()); lvs.append(lv.cpu()); ys.append(y); cs.append(c)
    mu_log = torch.cat(mus)
    lv = torch.cat(lvs)
    y = torch.cat(ys).numpy() * RUL_SCALE
    c = torch.cat(cs).numpy().astype(bool)
    # back to days; the log-normal median is the natural point estimate for MAE
    mu = (from_log_target(mu_log).numpy() * RUL_SCALE)
    sigma_log = torch.exp(0.5 * lv).numpy()
    hi = (from_log_target(mu_log + 1.96 * torch.exp(0.5 * lv)).numpy() * RUL_SCALE)
    lo = (from_log_target(mu_log - 1.96 * torch.exp(0.5 * lv)).numpy() * RUL_SCALE)
    sd = (hi - lo) / (2 * 1.96)

    obs = ~c
    if obs.sum() == 0:
        return {"n_observed": 0}
    err = mu[obs] - y[obs]
    ss = ((y[obs] - y[obs].mean()) ** 2).sum()
    within = (y[obs] >= lo[obs]) & (y[obs] <= hi[obs])
    return {
        "n_observed": int(obs.sum()),
        "mae_days": float(np.abs(err).mean()),
        "rmse_days": float(np.sqrt((err ** 2).mean())),
        "bias_days": float(err.mean()),
        "r2": float(1.0 - (err ** 2).sum() / ss) if ss > 0 else float("nan"),
        # a censored battery should be predicted to outlive its lower bound
        "censored_respect": (float((mu[c] >= y[c]).mean()) if c.any() else float("nan")),
        "picp_95": float(within.mean()),
        "mean_sigma_days": float(sd[obs].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LeadAcidBiMamba")
    ap.add_argument("--daily", type=Path, default=HERE / "daily.parquet")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--censor-weight", type=float, default=1.0)
    ap.add_argument("--logvar-l2", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=8,
                    help="epochs of plain MSE before enabling the NLL")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "checkpoint_lead_acid_v2.pt")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")

    print("building sequences …")
    tr, va, te, feats, norm = build(args.daily, seed=args.seed)
    verify_no_leakage(tr, va, te)
    print(f"  train {tr.summary()}")
    print(f"  val   {va.summary()}")
    print(f"  test  {te.summary()}")
    print(f"  {len(feats)} features, {tr.x.shape[1]}-day windows")

    # Sized against a 4 GB GTX 1650 Ti: the selective scan is sequential over
    # the window, so the batch matters more than depth here — 2 blocks at
    # batch 512 trains 4x faster than 3 blocks at batch 128 for the same memory.
    model = LeadAcidBiMamba(n_features=len(feats), d_model=args.d_model,
                            n_blocks=args.blocks, d_state=args.d_state,
                            seq_len=tr.x.shape[1],
                            dropout=args.dropout).to(device)
    print(f"\ndevice {device} | {count_parameters(model):,} parameters")

    tr_dl = make_loader(tr, args.batch, True)
    va_dl = make_loader(va, args.batch, False)
    te_dl = make_loader(te, args.batch, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr_dl), pct_start=0.25)

    best, best_epoch, bad = float("inf"), -1, 0
    history = []
    print(f"\n{'ep':>3s} {'train':>9s} {'val MAE':>9s} {'val RMSE':>9s} "
          f"{'val R2':>8s} {'PICP95':>7s} {'cens':>6s} {'s':>5s}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, c in tr_dl:
            x, y, c = x.to(device), y.to(device), c.to(device)
            opt.zero_grad(set_to_none=True)
            mu, lv = model(x)
            loss = loss_fn(mu, lv, to_log_target(y), c,
                           censor_weight=args.censor_weight,
                           logvar_l2=args.logvar_l2,
                           use_nll=(ep > args.warmup))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y); n += len(y)

        m = evaluate(model, va_dl, device)
        history.append({"epoch": ep, "train_loss": tot / n, **m})
        print(f"{ep:3d} {tot/n:9.4f} {m['mae_days']:9.1f} {m['rmse_days']:9.1f} "
              f"{m['r2']:8.3f} {m['picp_95']:7.3f} {m['censored_respect']:6.3f} "
              f"{time.time()-t0:5.1f}")

        if m["mae_days"] < best - 0.5:
            best, best_epoch, bad = m["mae_days"], ep, 0
            torch.save({"state_dict": model.state_dict(),
                        "features": feats, "norm": norm,
                        "config": {"d_model": args.d_model, "n_blocks": args.blocks,
                                   "d_state": args.d_state,
                                   "dropout": args.dropout,
                                   "seq_len": int(tr.x.shape[1])},
                        "val": m, "epoch": ep}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop — no improvement for {args.patience} epochs")
                break

    ck = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"\nbest epoch {best_epoch} (val MAE {best:.1f} d)")

    print("\n" + "=" * 68)
    print("HELD-OUT TEST — batteries never seen in training or validation")
    print("=" * 68)
    t = evaluate(model, te_dl, device)
    for k, v in t.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

    naive = np.abs(te.y[te.censored == 0] * RUL_SCALE
                   - (tr.y[tr.censored == 0] * RUL_SCALE).mean()).mean()
    print(f"\n  {'baseline (train mean)':20s} {naive:.1f} d MAE")
    print(f"  {'improvement':20s} {(1 - t['mae_days']/naive):.1%}")

    (args.out.parent / "history.json").write_text(json.dumps(
        {"history": history, "test": t, "baseline_mae_days": float(naive)}, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
