# Survival Analysis Outperforms Regression for Field Prognostics of VRLA Batteries

Code, logged results and figures for the preprint:

> **Survival Analysis Outperforms Regression for Field Prognostics of Valve-Regulated
> Lead-Acid Batteries: A Reproducible Benchmark on 1,027 Solar Off-Grid Systems**
> Dikshant — submitted to Research Square, 2026-08-05.
> ORCID [0009-0006-9434-610X](https://orcid.org/0009-0006-9434-610X)

The manuscript as submitted is in [`paper/`](paper/).

---

## What this is

Prognostics research is overwhelmingly lithium-ion, while valve-regulated lead-acid still
dominates stationary storage across much of the developing world. A 2023 systematic mapping
study of lead-acid prognostics found **no public dataset**, so every reported error came
from a private cohort of 3–70 cells and none were comparable.

The Oxford/BBOXX release of **1,027 field-deployed 12 V/20 Ah VRLA batteries** on solar home
systems in sub-Saharan Africa closes that gap, and this repository is a reproducible
benchmark built on it.

**Central methodological finding:** the task must be posed as **survival analysis, not
regression**. 52 % of units are right-censored. A bidirectional Mamba regressor trained on
remaining-useful-life in days reaches 83.3 d MAE — **worse than a one-line `mean_lifetime −
age` heuristic (63.8 d) and worse than a ridge regression fitted in under a second
(57.2 d)**. Re-posed as landmark survival analysis, gradient-boosted survival reaches
**AUC 0.848 (95 % CI 0.819–0.877)** at four weeks.

---

## Headline results, and where each number comes from

Every figure in the paper traces to a file in [`results/`](results/). Nothing is restated
from a draft.

| claim | file | key |
|---|---|---|
| AUC 0.848 @4wk, CI [0.819, 0.877] | `paper_log.txt` | the `HEADLINE` block |
| AUC 0.832 @12wk, CI [0.808, 0.855] | `paper_log.txt` | same |
| Uno's IPCW C 0.737–0.756 (5 estimators) | `paper_results.json` | `table1.*.at_risk.uno_c` |
| 84.1 % @8wk / 89.7 % at failure | `aitio_comparison.json` | vs published 73 % / 82 % |
| age-only ≈ random (AUC 0.545) | `paper_results.json` | `table2_ablation.*.age only` |
| 6,003 landmarks from 1,026 of 1,027 cells | `survival_comparison.json` | `n_rows`, `n_batteries` |

The confidence intervals are bootstrapped **over batteries, not rows** (`boot_ci` in
`src/paper_results.py`) — resampling rows would leak the same battery across the resample
and understate the interval.

---

## Reproducing

```bash
pip install numpy pandas scikit-learn scikit-survival pyarrow matplotlib torch

python src/build_daily.py        # raw 60 s telemetry -> daily.parquet   (see Data, below)
python src/survival_data.py      # landmark construction, 97 features
python src/compare_survival.py   # Table 1: five survival models
python src/replicate_aitio.py    # Table 5: head-to-head on the published protocol
python src/paper_results.py --seed 7   # headline + ablation + calibration
```

Splits are deterministic **at the battery level** — no battery appears in more than one
fold. `--seed 7` reproduces the published numbers.

---

## Data — not included here

The battery data is the Oxford University Research Archive release accompanying Aitio &
Howey (2021):

- **DOI:** [`10.5287/bodleian:aVR4oDV4N`](https://doi.org/10.5287/bodleian:aVR4oDV4N)
- **Licence: CC BY-NC 4.0**

**No raw or derived data is redistributed in this repository.** That includes
`daily.parquet` (62 MB), which is derived from the release and therefore carries the same
licence. Download the source data from the DOI above and regenerate it with
`src/build_daily.py`.

This work is non-commercial academic research and complies with CC BY-NC. If you intend a
commercial application, the licence does not permit it.

---

## Also not included

**Model checkpoints** (`checkpoint_*.pt`). These are weights for the bidirectional Mamba
RUL regressor, which is reported in the paper as a **negative result** — 83.3 d MAE, beaten
by a ridge regression at 57.2 d. Shipping them would invite someone to treat a failed model
as a deliverable. `src/train.py` regenerates them if you want to confirm the negative result
yourself.

---

## Known issue in the submitted manuscript

One sentence in §6 of the submitted preprint is **not supported by its citation**:

> "the linear Cox proportional hazards model matches or exceeds the random survival forest,
> **reversing the ordering reported by Voronov et al.**"

Voronov et al. (*IEEE Transactions on Reliability* **67**(2), 623–639, 2018) never fitted a
Cox model. Their paper states: *"a straightforward application of the Cox regression model
is not applicable and motivates our choice of the non-parametric RSF model."* There is no
ordering to reverse. In Frisk et al. (PHM 2014) "Cox" appears only as a textbook citation.

The finding itself stands — Cox does match or exceed RSF on this dataset (`paper_results.json`,
`table1`). Only the comparison to prior work is unsupported, and it will be removed at
journal submission.

### Where that error came from

Auditing the code for this release found the source. Two files printed **fabricated
reference points**:

```
src/paper_results.py   "Voronov & Frisk (Scania fleet, lead-acid, RSF)   AUC 0.69 – 0.772"
src/compare_survival.py  same line, plus
                       "Voronov & Frisk (same study, Cox regression)     AUC 0.63 – 0.675"
```

**Neither range appears in any of the three Linköping papers**, all of which were obtained
and read in full for this audit:

- **Voronov et al.**, *IEEE T-Rel* **67**(2), 623–639 (2018) — reports error rate and
  concordance, **no AUC at all**, and states Cox is *"not applicable"*
- **Frisk, Krysander & Larsson**, PHM Society (2014) — uses AUC only as a
  *variable-importance* score, not a predictive metric
- **Frisk & Krysander**, *IFAC-PapersOnLine* **48**(21), 105–112 (2015) — reports neither

The second invented line is what made the manuscript's "reversing the ordering" claim look
supported: it fabricated a Cox result for a paper that never fitted Cox. Both lines are
removed here, with comments marking where they were and why.

---

## Layout

```
src/        pipeline, in run order (see Reproducing)
results/    every JSON and log the paper's numbers are drawn from
paper/      manuscript as submitted (.docx + .pdf) and the 7 figures
```

## Licence

Code: MIT (see `LICENSE`). Data: not included; CC BY-NC 4.0 at the source DOI.
