# NSUCEC Datathon — FictiPay (Team Cybernauts)

## Rounds
- **Round 1 (won, #1):** binary churn. Built by `proto/build_notebook.py` (now superseded as the
  active `notebook.ipynb`). Exploited a TrxID sequential-block leak → AUC ~0.999.
- **Round 2 (current FINAL):** **survival** — predict *when* a customer churns. Active
  `notebook.ipynb` is built by **`proto/build_survival_notebook.py`**.

## What's here
| Path | Purpose |
|---|---|
| `notebook.ipynb` | **The Round-2 deliverable.** Kaggle-runnable survival pipeline; generates `predictions.csv` (5-col) + features.md/report.pdf/presentation.pdf/explainability/. |
| `public/` | Round-2 (re-simulated) data. `public_prev/` = Round-1 data. |
| `proto/build_survival_notebook.py` | **Generates `notebook.ipynb`** — edit here, then `python3 proto/build_survival_notebook.py`. |
| `survival_analysis_guide.pdf` / `.tex` | Plain-English survival-analysis study guide. |
| `survival_cookbook.ipynb` | Runnable copy-paste survival recipes (built by `proto/build_survival_cookbook.py`). |

## The Round-2 problem
Churn = **30 consecutive silent days**. Survival time `T` measured from **Mar 31**; right-censored
at **Jun 30** (`T=91`). Only **Jan–Mar** data given. Submit per test customer:
`ACCOUNT_ID, RISK_SCORE, SURV_PROB_30D, SURV_PROB_60D, SURV_PROB_90D` (probs monotone ↓).
**Scored:** c-index on `RISK_SCORE` (40 pts) + Integrated Brier Score on `SURV_PROB_*` (30 pts).
Top-5 present (30 pts).

## Key findings (this round)
- **The leak is patched.** TrxIDs are now random 56-bit hex (`T-…`): corr(time-order, id)=0.003,
  within-customer monotonic fraction 0.50 → no sequential/block structure. Stage 4b audits this in
  the notebook (good Phase-2 talking point). Predictions are fully model-based.
- **Re-simulated dataset:** only ~70% ID overlap with Round 1; old/new labels don't map. Event rate
  **5.2%**; events cluster at 70–90 days; `EVENT=0 ⟹ DURATION=91`.
- **Modeling simplification:** all scored horizons (30/60/90) ≤ censor time 91, so `SURV_PROB`
  reduces to **3 fully-observed binary classifiers** `S_h = 1[T>h]` — no censoring math for IBS.
- **Strongest signals:** recency, `gap_z` (personal rhythm), 30-day activity, balance drain.

## Model (notebook Stage 5)
- 3 LightGBM classifiers → `SURV_PROB_30/60/90` (isotonic-calibrated, monotonicity-enforced).
- XGBoost `survival:cox` → continuous `RISK_SCORE`; final risk is the OOF-best of {cox, 1−S90,
  Σ(1−S_h), cox+surv blend}.
- lifelines Cox (hazard ratios) + sksurv Random Survival Forest (curves) for Phase-2 interpretation.

## Local FAST smoke result (80 K subsample, 3 folds, 8 trials)
OOF **c-index 0.749**, **IBS 0.0133** → est. **~42/70**. Full run (595 K, 5 folds, 35 trials) should
land higher. `predictions.csv`: 255 000 rows, 5 cols, monotone, validated.

## Running on Kaggle
1. Create a Kaggle Dataset from `public/` (keep `transactions/` and `dayend_balance/` subfolders).
2. New Notebook → upload `notebook.ipynb` → attach dataset → **enable Internet** (pip installs
   duckdb / lifelines / scikit-survival) → Run All.
3. Outputs in `/kaggle/working/`: `predictions.csv`, `features.md`, `report.pdf`,
   `presentation.pdf`, `explainability/`, `figures/`.

Approx. full runtime: ~1–1.5 h on Kaggle CPU. Set env `FAST=1` for a ~5-min subsampled debug run.
