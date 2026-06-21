# FictiPay Round 2 — Survival Modelling Approach (Team Cybernauts)

A complete, plain-language record of what the pipeline (`notebook.ipynb`, built by
`proto/build_survival_notebook.py`) does and *why* at every step. Doubles as the source material
for the Phase-2 presentation.

---

## 1. The problem in one paragraph

Round 1 asked **whether** a customer churns. Round 2 asks **when**. A customer "churns" on the day
they begin **30 consecutive days with no transaction**. We measure survival time **T** in days from
the **31 March 2024** cutoff; anyone still active on **30 June 2024** is **right-censored** at
T = 91. We only get **January–March 2024** data and must predict, for each test customer:

| Column | Meaning |
|---|---|
| `RISK_SCORE` | any monotone churn-risk score (higher = churns sooner) — scored by **c-index** |
| `SURV_PROB_30D` | P(still active at day 30) = P(T > 30) |
| `SURV_PROB_60D` | P(T > 60) |
| `SURV_PROB_90D` | P(T > 90) — scored, with 30/60, by **Integrated Brier Score** |

Hard rule: `SURV_PROB_30D ≥ SURV_PROB_60D ≥ SURV_PROB_90D` (monotone non-increasing).

**Scoring (Phase 1, 70 pts):**
- c-index: `max(0, (c − 0.5) / 0.5) × 40`
- IBS: `max(0, (0.05 − IBS) / 0.05) × 30`

Top-5 teams by Phase-1 score present (Phase 2, 30 pts).

---

## 2. Dataset analysis (what we found before modelling)

**It is a re-simulated dataset, not Round 1 extended.** Only ~70% of customer IDs overlap with
Round 1 and the old/new labels don't map, so we treat `public/` as completely fresh.

| Fact | Value | Why it matters |
|---|---|---|
| Train rows | 595,000 (`ACCOUNT_ID, DURATION_DAYS, EVENT_FLAG`) | the survival labels |
| Test rows | 255,000 (`ACCOUNT_ID` only) | what we predict |
| Event rate | **5.2%** | matches the IBS "~5% baseline"; heavily imbalanced |
| Event timing | clusters at **70–90 days** | most churners go quiet *late*, not early |
| Censoring | `EVENT=0 ⟹ DURATION=91` always | clean, single censoring time |
| Transactions | 80.1 M rows, Jan–Mar only | too big for RAM → DuckDB |
| Balances | 77.35 M rows, Jan–Mar only | balance-trajectory covariates |
| Recency signal | event rate climbs **4.3% → 17%** as recency goes 0→40 d | strong but **not** deterministic |

A tempting shortcut — "silent ≥30 days at the cutoff ⟹ already churned" — **does not hold**: those
customers are mostly censored because they *resumed* activity in Apr–Jun. So this is a genuine
probabilistic survival problem; recency is a strong covariate, not an oracle.

### 2.1 The leakage audit (Round-1 exploit is patched)

Round 1 had a structural leak: `TrxID`s were **sequential integers** in contiguous per-customer
blocks, so the gap to the next block encoded a customer's hidden post-March activity. We re-audited
the new data and the leak is **gone**:

| Test | Round 1 | Round 2 |
|---|---|---|
| TrxID format | sequential integer | **random hex** (`T-E7B27344F769E9`) |
| corr(time-order, ID value) | ~1.0 | **0.003** |
| within-customer monotonic w/ time | True | **0.498** (random) |
| usable for hidden-activity inference | yes | **no** |

The organisers randomised the IDs specifically to close our Round-1 exploit. **All predictions this
round are fully model-based and legitimate** — and the notebook documents this audit (Stage 4b) as a
Phase-2 talking point ("we checked, it's patched").

### 2.2 The key modelling simplification

All three scored horizons (30/60/90) are **≤ the censoring time (91)**. A censored customer is known
to be active through day 91, so their status at 30/60/90 is **fully known**. Therefore each
`SURV_PROB` reduces to a **fully-observed binary label** `S_h = 1[T > h]` — **no censoring
correction is needed** for the probability part. This lets us aim our strong gradient-boosting
machinery directly at the Brier/IBS metric, while still using a proper survival model for the
ranking (`RISK_SCORE` / c-index).

---

## 3. Pipeline, stage by stage

### Stage 0 — Setup
Installs (`duckdb, lightgbm, xgboost, optuna, shap, lifelines, scikit-survival`); robust Kaggle
data-path detection (explicit competition mounts → recursive search ignoring any `prev` copy →
local fallback); a `FAST` switch for a ~5-min subsampled debug run; constants
(`HORIZONS=[30,60,90]`, `CENSOR_T=91`, cutoff = 1 Apr, observation = Jan–Mar).

### Stage 1 — Large-scale data handling (DuckDB)
**DuckDB** is a vectorised, multi-threaded, **out-of-core** SQL engine — it streams aggregations over
the 80 M-row Parquet files and spills to disk, beating Spark (cluster overhead) and Dask (slower
task graph) on single-node aggregation. We use column pruning + predicate pushdown; only the final
per-customer feature matrix (~595 K × 138) enters pandas. A **leakage guard** asserts every record
falls inside Jan–Mar before any feature is built. EDA uses reservoir sampling (`USING SAMPLE`).

### Stage 2 — Feature engineering (138 survival covariates)
The rich behavioural pipeline from Round 1 carries over verbatim — it is exactly the set of
covariates a survival model needs. Six families:

1. **Outgoing activity** — counts/amounts over nested windows (3/7/14/30/60/90 d), per-type counts
   and recency, active days/weeks, monthly counts, time-of-day mix.
2. **Personal rhythm (survival-style)** — inter-transaction gap stats; `gap_z` = (current silence −
   personal mean gap) / personal std ("how abnormal is *this* customer's silence?"); silent-window
   share; a Poisson zero-activity probability.
3. **Trajectory** — 13 weekly counts + slope + trailing-silent-weeks (the *shape* of decay).
4. **Incoming & network** — P2P received, recurring counterparties, peer-graph liveness ("are this
   customer's contacts still active?").
5. **Balance dynamics** — level, volatility, slope, zero-balance days, days since last change, drain
   ratio (a draining wallet precedes exit).
6. **Profile (KYC)** — tenure (open date may pre-date 2024), region, gender.

All are computed strictly from Jan–Mar. The annotated list is written to `features.md`.

### Stage 3 — Feature quality & Pareto handling
Fintech data is Pareto (a few power-users dominate volume) and zero-inflated. We quantify skew &
zero-share, **winsorize** heavy tails at p99.9, keep **log1p** copies for the linear/Cox model, and
add **indicator flags** so models separate "structurally zero" from "low but active". Trees and RSF
are invariant to monotone transforms, so raw columns feed the GBMs.

### Stage 4 — Target + leakage audit
Builds the fully-observed survival labels `surv_h = 1[T > h]`, reports base survival rates
(S30 ≈ 0.9995, S60 ≈ 0.997, S90 ≈ 0.948), plots the event-time histogram and the population
Kaplan–Meier curve, and runs the TrxID randomisation audit described in §2.1.

### Stage 5 — Survival models (the core)
A **5-fold stratified out-of-fold (OOF)** loop (stratified on `EVENT_FLAG`). In each fold we train:

- **Three LightGBM classifiers** → `SURV_PROB_30/60/90`. Labels are fully observed (§2.2), so these
  directly optimise calibration / Brier. After CV we apply **isotonic calibration** (fit on OOF,
  applied to test) and **enforce monotonicity** `S30 ≥ S60 ≥ S90` via a running minimum.
- **XGBoost `survival:cox`** → a continuous, finely-resolved risk score that ranks the whole
  timeline (the partial-likelihood objective is exactly the c-index's target). Censoring is encoded
  by the label sign (positive time = event, negative = censored).

We then choose the `RISK_SCORE` by **OOF c-index** among four candidates — raw XGBoost-Cox,
`1 − S90`, `Σ(1 − S_h)`, and a rank-blend of Cox + survival probs — and keep the winner.

**Why this split?** The two scored metrics reward different things: IBS rewards calibrated
probabilities (→ dedicated classifiers), c-index rewards fine ranking across times (→ a survival
model). Optimising each with the tool best suited to it beats forcing one model to do both.

### Stage 6 — Hyperparameter tuning (Optuna TPE)
We tune the **90-day classifier** — the hardest horizon, where almost all events live and the
biggest lever on both metrics. Tree-structured Parzen Estimator searches where the surrogate expects
improvement (far more sample-efficient than grid search); a convergence chart is saved.

### Stage 7 — Explainability (for Phase 2)
- **SHAP** on the S(90) classifier → which behaviours drive surviving the quarter, in plain language.
- **Cox hazard ratios (lifelines)** on a stable 10-covariate standardised model → interpretable
  per-1-SD multipliers ("recency HR ≈ 1.24 → each SD of extra silence raises the churn rate ~24%").
- **Random Survival Forest curves (scikit-survival)** → visual divergence of predicted survival
  across low/median/high-risk segments, with the 30/60/90 markers.

### Stage 8 — Business recommendations (time-sensitive retention)
We bucket customers by predicted 30-day churn probability into `watch / soon / urgent / critical`
tiers, each mapped to a **timed** intervention (the hazard says *how urgent*), and quantify the
top-decile lift (targeting the riskiest 10% captures a multiple of the base churn rate).

### Submission + scoring
Writes `predictions.csv` (the exact 5 columns), with guarantees enforced: probabilities in [0,1],
monotone non-increasing, `RISK_SCORE` finite and non-negative, all 255 K test IDs present and
unique. Then prints the OOF **c-index** and **IBS** and the estimated Phase-1 score, and generates
`report.pdf` + `presentation.pdf` + the `explainability/` plots.

---

## 4. Evaluation methodology

- **5-fold stratified OOF** (on `EVENT_FLAG`) — every training prediction comes from a model that
  never saw that row, so reported numbers are honest, not in-sample.
- Metrics are the **exact competition metrics**: Harrell's **c-index** (`concordance_index_censored`)
  on `RISK_SCORE`, and **Integrated Brier Score** (`integrated_brier_score`, with a pure-Brier
  fallback) on the survival-probability matrix at [30, 60, 90].
- The notebook converts both to the official points formula so the printed "≈ X / 70" matches how
  the leaderboard scores us.

---

## 5. Results (local FAST smoke run)

FAST mode = 80 K-row subsample, 3 folds, 8 Optuna trials (a quick correctness check, **not** the
real run):

| Metric | Value | Points |
|---|---|---|
| c-index (RISK_SCORE) | **0.749** | 19.9 / 40 |
| IBS (SURV_PROB) | **0.0133** | 22.0 / 30 |
| **Phase-1 total** | — | **≈ 41.9 / 70** |

- Brier @30/60/90 = 0.0005 / 0.0032 / 0.0466 (nearly all error sits at the 90-day horizon, as
  expected — that's where the events are).
- Chosen `RISK_SCORE`: the Cox + survival rank-blend (beat the other three candidates on OOF c-index).
- Leak audit: **no leak**.

**The full Kaggle run** (all 595 K rows, 5 folds, 35 Optuna trials) should land **higher**,
especially on c-index — more data and deeper tuning help ranking most. Estimate ~45–48/70; the
running notebook will print the honest figure.

---

## 6. Why these choices (defence for Q&A)

- **Why not one survival model for everything?** The two metrics pull in different directions
  (calibration vs ranking). Three calibrated classifiers nail the probability metric; a Cox-style
  ranker nails the ranking metric. Specialising beats compromising.
- **Why classifiers are valid here despite censoring** — because every scored horizon is at/under
  the single censoring time (91), so the binary labels are fully observed (§2.2). This is a property
  of *this* problem, not a general shortcut.
- **Why DuckDB** — single-node, out-of-core, fastest for wide group-bys at this scale; no cluster.
- **Why we don't exploit any "leak"** — there isn't one anymore; we audited and the IDs are
  randomised. Our edge is the feature pipeline and the metric-aware model split.
- **Interpretability** — Cox hazard ratios + SHAP give plain-language drivers (recency, personal
  rhythm `gap_z`, 30-day activity, balance drain) for the Phase-2 business story.

---

## 7. How to run

1. Attach the Kaggle dataset (`…/bkash-presents-nsucec-datathon-final/public`).
2. Upload `notebook.ipynb`, **enable Internet** (for the `scikit-survival` / `lifelines` installs),
   **Run All** (~1–1.5 h on Kaggle CPU; `FAST=1` for a ~5-min debug run).
3. Submit the generated `predictions.csv`.

To edit the pipeline, change `proto/build_survival_notebook.py` and run
`python3 proto/build_survival_notebook.py` to regenerate `notebook.ipynb`.

---

## 8. Artifacts produced

`predictions.csv` (submission) · `features.md` (covariate catalogue) · `report.pdf` (metrics +
plots) · `presentation.pdf` (5-slide deck) · `explainability/` (SHAP summary, Cox hazard-ratio
plot, RSF survival curves) · `figures/` (target, Optuna, tiers).
