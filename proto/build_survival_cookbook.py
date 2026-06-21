"""Builds survival_cookbook.ipynb — copy-paste survival-analysis recipes for the
NSUCEC Datathon final round (no-AI conditions). Run: python3 proto/build_survival_cookbook.py
Edit THIS file, then regenerate the notebook.
"""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})

# ============================================================ title
md(r"""# Survival Analysis — Datathon Cookbook (no-AI ready)
**Team Cybernauts** · copy-paste recipes for every survival technique.

Each section is **self-contained**: it runs against the synthetic dataset built in §0, so you can
lift any block and swap in your own data. Order of use in a real problem:

| Step | Section | Recipe |
|---|---|---|
| Frame the data | §1 | Build `time` + `event`, split, make `Surv` arrays |
| Explore | §2 | Kaplan–Meier curves + log-rank group test |
| Interpret factors | §3 | Cox model → hazard ratios + PH assumption check |
| Shape / forecast | §4 | Weibull & AFT, pick distribution by AIC |
| ML model | §5–6 | Random Survival Forest, Gradient Boosting, XGBoost-AFT |
| Score it | §7 | C-index, Integrated Brier, time-dependent AUC, CV harness |
| Fixed-horizon | §8 | Convert survival → "event by time H?" classification (your churn case) |
| Cram | §9 | One-line cheat cells |

> Replace `make_survival_data()` in §0 with your own loader. Everything downstream keys off the
> column names **`time`** (duration) and **`event`** (1 = happened, 0 = censored).
""")

# ============================================================ 0 setup
md(r"""## §0 — Setup, imports, and a synthetic dataset

Run this once. The installs are no-ops if the packages are already present (e.g. on Kaggle).""")

code(r"""# --- installs (safe to re-run; comment out if offline & already installed) ---
import sys, subprocess
for pkg in ['lifelines', 'scikit-survival', 'xgboost']:
    try:
        __import__('sksurv' if pkg == 'scikit-survival' else pkg.replace('-', '_'))
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import os; os.makedirs('figures', exist_ok=True)
SEED = 42; np.random.seed(SEED)

import lifelines, sksurv, xgboost as xgb
print('lifelines', lifelines.__version__, '| sksurv', sksurv.__version__, '| xgboost', xgb.__version__)
""")

code(r'''# --- synthetic time-to-event data with RIGHT-CENSORING (replace with your own) ---
def make_survival_data(n=2000, seed=SEED):
    """Returns a DataFrame with covariates + `time` (duration) + `event` (1=event, 0=censored).
    Times are Weibull-distributed; risk rises with age, falls with balance, group=1 is riskier."""
    rng = np.random.default_rng(seed)
    age     = rng.normal(50, 12, n).clip(18, 90)
    balance = rng.exponential(1000, n)
    region  = rng.integers(0, 3, n)              # categorical 0/1/2
    group   = rng.integers(0, 2, n)              # a binary group for log-rank demos
    # linear predictor: higher -> event happens SOONER
    lp = 0.03 * (age - 50) - 0.0005 * (balance - 1000) + 0.7 * group + 0.2 * region
    shape_k   = 1.3                              # Weibull shape of the TRUE process
    scale     = np.exp(2.6 - lp)                 # covariates accelerate/decelerate the clock
    event_time  = scale * rng.weibull(shape_k, n)
    censor_time = rng.uniform(0, np.percentile(event_time, 85), n)   # random + admin censoring
    time  = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({'age': age, 'balance': balance, 'region': region, 'group': group,
                         'time': time.round(3), 'event': event})

df = make_survival_data()
print(df.head())
print(f"\nn={len(df)}  event rate={df.event.mean():.1%}  censoring rate={1-df.event.mean():.1%}")
print(f"follow-up time: min={df.time.min():.2f}  median={df.time.median():.2f}  max={df.time.max():.2f}")
''')

# ============================================================ 1 data prep
md(r"""## §1 — Data prep: durations, events, split, `Surv` arrays

The two libraries want the target in **different shapes**:
* **lifelines** → a DataFrame with a duration column and an event column.
* **scikit-survival** → a structured array `y` built with `Surv.from_arrays(event_bool, time)`.

`safe_times()` returns evaluation time-points that lie strictly inside *both* the train and test
event ranges — required by the Brier / time-dependent-AUC metrics (they error outside the range).""")

code(r'''from sklearn.model_selection import train_test_split
from sksurv.util import Surv

FEATURES = ['age', 'balance', 'region', 'group']
X = df[FEATURES].astype(float)
t = df['time'].values
e = df['event'].values

X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
    X, t, e, test_size=0.25, random_state=SEED, stratify=e)

# scikit-survival structured targets (event must be boolean)
y_tr = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
y_te = Surv.from_arrays(event=e_te.astype(bool), time=t_te)

# lifelines-style frames (duration + event columns side by side with features)
df_tr = X_tr.assign(time=t_tr, event=e_tr)
df_te = X_te.assign(time=t_te, event=e_te)

def safe_times(n=10):
    """Evaluation times strictly inside both train & test EVENT ranges (for Brier / AUC)."""
    lo = max(t_tr[e_tr == 1].min(), t_te[e_te == 1].min())
    hi = min(t_tr[e_tr == 1].max(), t_te[e_te == 1].max())
    return np.linspace(lo, hi, n + 2)[1:-1]

EVAL_TIMES = safe_times()
print('train', X_tr.shape, '| test', X_te.shape)
print('eval times:', np.round(EVAL_TIMES, 2))
''')

# ============================================================ 2 KM + logrank
md(r"""## §2 — Kaplan–Meier curves + log-rank test

**KM** = the "% still alive over time" staircase (no model, no assumptions). **Log-rank** = a
yes/no test of whether two (or more) groups' curves really differ.""")

code(r'''from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# --- overall KM curve + median ---
kmf = KaplanMeierFitter()
kmf.fit(durations=df['time'], event_observed=df['event'], label='all customers')
print('median survival time:', kmf.median_survival_time_)
print('S(t) at t=5,10,20:\n', kmf.survival_function_at_times([5, 10, 20]))

# --- KM by group (overlay) + plot to file ---
fig, ax = plt.subplots(figsize=(7, 4))
for g, sub in df.groupby('group'):
    KaplanMeierFitter().fit(sub['time'], sub['event'], label=f'group={g}').plot_survival_function(ax=ax)
ax.set_xlabel('time'); ax.set_ylabel('S(t) = fraction still active'); ax.set_title('Kaplan–Meier by group')
plt.tight_layout(); plt.savefig('figures/km_by_group.png', dpi=130); plt.show()

# --- log-rank: two groups ---
a, b = df[df.group == 0], df[df.group == 1]
lr = logrank_test(a['time'], b['time'], a['event'], b['event'])
print(f'\nlog-rank group0 vs group1:  p={lr.p_value:.2e}  stat={lr.test_statistic:.2f}')

# --- log-rank: more than two groups (e.g. region 0/1/2) ---
mlr = multivariate_logrank_test(df['time'], df['region'], df['event'])
print(f'multivariate log-rank across regions:  p={mlr.p_value:.2e}  stat={mlr.test_statistic:.2f}')
''')

# ============================================================ 3 Cox
md(r"""## §3 — Cox Proportional Hazards (hazard ratios + assumption check)

Cox gives an interpretable **hazard ratio (HR = `exp(coef)`)** per feature: `>1` riskier (event
sooner), `<1` protective. Then you **must** check the proportional-hazards (PH) assumption.""")

code(r'''from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df_tr, duration_col='time', event_col='event')
cph.print_summary(columns=['coef', 'exp(coef)', 'p'])   # exp(coef) == hazard ratio

# tidy HR table sorted by strength
hr = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
hr = hr.rename(columns={'exp(coef)': 'HR'}).sort_values('HR', ascending=False)
print('\nHazard ratios (>1 = riskier):\n', hr.round(3))

# forest plot of HRs
ax = cph.plot(); ax.set_title('Cox hazard ratios (log scale)')
plt.tight_layout(); plt.savefig('figures/cox_forest.png', dpi=130); plt.show()

# predictions for new customers
print('\nrisk score exp(beta.x) (higher=riskier):\n', cph.predict_partial_hazard(X_te.head(3)).round(3).values.ravel())
print('predicted median survival time:\n', cph.predict_median(X_te.head(3)).values.ravel())
''')

code(r'''# --- PH assumption check: Schoenfeld residuals ---
from lifelines.statistics import proportional_hazard_test

ph = proportional_hazard_test(cph, df_tr, time_transform='rank')
ph.print_summary()   # small p for a feature => its effect drifts over time => PH VIOLATED
# graphical version (saves residual-vs-time plots):
# cph.check_assumptions(df_tr, p_value_threshold=0.05, show_plots=True)

# --- FIX 1: stratify on the offending feature (separate baseline per stratum, lose its HR) ---
cph_strat = CoxPHFitter().fit(df_tr, 'time', 'event', strata=['region'])
print('\nstratified model fitted; concordance =', round(cph_strat.concordance_index_, 4))

# --- FIX 2 (skeleton): time-varying effect — add a feature x log(time) interaction in long format,
#     then use lifelines.CoxTimeVaryingFitter on the start/stop dataset. ---
''')

# ============================================================ 4 parametric / AFT
md(r"""## §4 — Parametric & AFT models (Weibull, shape `k`, pick by AIC)

Parametric models commit to a hazard **shape** and can **extrapolate beyond the data**. The Weibull
**shape `rho_` (= `k`)** is the headline: `>1` risk rises, `=1` flat (exponential), `<1` risk falls.
Compare candidate distributions by **AIC (lower = better)**.""")

code(r'''from lifelines import (WeibullFitter, WeibullAFTFitter,
                       LogNormalAFTFitter, LogLogisticAFTFitter, ExponentialFitter)

# --- univariate Weibull: read the shape ---
wf = WeibullFitter().fit(df['time'], df['event'])
print(f'Weibull shape k (rho_) = {wf.rho_:.3f}  ->',
      'risk RISES (k>1)' if wf.rho_ > 1 else ('FLAT (k=1)' if abs(wf.rho_-1) < 0.05 else 'risk FALLS (k<1)'))
print(f'Weibull scale (lambda_) = {wf.lambda_:.3f}   AIC = {wf.AIC_:.1f}')

# --- AFT regression: covariates stretch/shrink the time clock; compare distributions by AIC ---
aft_models = {'Weibull': WeibullAFTFitter(), 'LogNormal': LogNormalAFTFitter(),
              'LogLogistic': LogLogisticAFTFitter()}
for name, m in aft_models.items():
    m.fit(df_tr, 'time', 'event')
    print(f'{name:12s} AIC={m.AIC_:.1f}  concordance={m.concordance_index_:.4f}')

best = min(aft_models.values(), key=lambda m: m.AIC_)
print('\nbest AFT by AIC:', type(best).__name__)
print('time ratios exp(coef) (>1 = event LATER / protective):')
print(best.summary['exp(coef)'].round(3))
print('\npredicted median time for 3 test customers:', best.predict_median(X_te.head(3)).values.ravel().round(2))
''')

code(r'''# --- the three Weibull hazard shapes (great viva visual) + extrapolation demo ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tt = np.linspace(0.1, 20, 200)
for k, lab in [(0.6, 'k<1 falling (early churn)'), (1.0, 'k=1 flat (exponential)'),
               (1.8, 'k>1 rising (wear-out)')]:
    h = (k / 5.0) * (tt / 5.0) ** (k - 1)        # Weibull hazard, scale=5
    axes[0].plot(tt, h, label=lab)
axes[0].set_title('Weibull hazard shapes'); axes[0].set_xlabel('time'); axes[0].set_ylabel('hazard h(t)'); axes[0].legend()

# extrapolation: parametric curve continues past the last observed time (Cox cannot)
horizon = np.linspace(0, df['time'].max() * 1.8, 200)
for i in range(3):
    sf = best.predict_survival_function(X_te.iloc[[i]], times=horizon)
    axes[1].plot(horizon, sf.values.ravel(), label=f'customer {i}')
axes[1].axvline(df['time'].max(), ls='--', c='grey', label='last observed time')
axes[1].set_title('AFT extrapolates beyond the data'); axes[1].set_xlabel('time'); axes[1].set_ylabel('S(t)'); axes[1].legend()
plt.tight_layout(); plt.savefig('figures/weibull_shapes.png', dpi=130); plt.show()
''')

# ============================================================ 5 RSF
md(r"""## §5 — Random Survival Forest (sklearn-style, no PH assumption)

A forest that predicts a **risk score** (higher = event sooner) and a full **survival curve** per
subject, capturing non-linearities and interactions automatically.""")

code(r'''from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                           max_features='sqrt', n_jobs=-1, random_state=SEED)
rsf.fit(X_tr, y_tr)
risk_rsf = rsf.predict(X_te)                       # higher = more risk
print('RSF train C-index:', round(rsf.score(X_tr, y_tr), 4))

# predicted survival curves for a few customers
# (each StepFunction carries its time grid on `.x`; values via fn(times))
surv_fns = rsf.predict_survival_function(X_te.iloc[:4])
times_rsf = surv_fns[0].x
fig, ax = plt.subplots(figsize=(7, 4))
for i, fn in enumerate(surv_fns):
    ax.step(times_rsf, fn(times_rsf), where='post', label=f'customer {i}')
ax.set_title('RSF predicted survival curves'); ax.set_xlabel('time'); ax.set_ylabel('S(t)'); ax.legend()
plt.tight_layout(); plt.savefig('figures/rsf_curves.png', dpi=130); plt.show()
''')

# ============================================================ 6 GBM / Coxnet / importance
md(r"""## §6 — Gradient Boosting, regularised Cox, and permutation importance

`GradientBoostingSurvivalAnalysis` and `CoxnetSurvivalAnalysis` (L1/L2-penalised Cox for many
features). Permutation importance works on any sksurv model via its built-in C-index `score`.""")

code(r'''from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

gbm = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.1,
                                       max_depth=3, subsample=0.8, random_state=SEED)
gbm.fit(X_tr, y_tr)
print('GBM survival train C-index:', round(gbm.score(X_tr, y_tr), 4))

# regularised Cox (good when p is large / collinear); needs scaling
coxnet = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
coxnet.fit(X_tr, y_tr)
print('Coxnet train C-index:', round(coxnet.score(X_tr, y_tr), 4))

# permutation importance (drop in C-index when a feature is shuffled)
imp = permutation_importance(rsf, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1)
imp_df = pd.DataFrame({'feature': FEATURES, 'importance': imp.importances_mean}
                      ).sort_values('importance', ascending=False)
print('\nRSF permutation importance (loss of C-index):\n', imp_df.round(4).to_string(index=False))
''')

# ============================================================ 7 XGBoost AFT
md(r"""## §7 — XGBoost Accelerated Failure Time (`survival:aft`)

Boosting that predicts the **actual event time** and ingests censoring via per-row **label bounds**:
uncensored → `lower = upper = time`; right-censored → `lower = time`, `upper = +inf`.""")

code(r'''import xgboost as xgb

d_tr, d_te = xgb.DMatrix(X_tr), xgb.DMatrix(X_te)
# right-censored encoding
lower_tr = t_tr.copy()
upper_tr = np.where(e_tr == 1, t_tr, np.inf)
d_tr.set_float_info('label_lower_bound', lower_tr)
d_tr.set_float_info('label_upper_bound', upper_tr)

params = dict(objective='survival:aft', eval_metric='aft-nloglik',
              aft_loss_distribution='normal',        # or 'logistic' / 'extreme'
              aft_loss_distribution_scale=1.0,
              tree_method='hist', learning_rate=0.05, max_depth=3,
              subsample=0.8, colsample_bytree=0.8)
bst = xgb.train(params, d_tr, num_boost_round=300)

pred_time = bst.predict(d_te)        # predicted event time: HIGHER = longer survival = LOWER risk
risk_xgb = -pred_time                 # flip sign to get a risk score for C-index

from sksurv.metrics import concordance_index_censored
c_xgb = concordance_index_censored(e_te.astype(bool), t_te, risk_xgb)[0]
print('XGBoost-AFT test C-index:', round(c_xgb, 4))
''')

# ============================================================ 8 evaluation
md(r"""## §8 — Evaluation: C-index, Integrated Brier, time-dependent AUC + CV harness

* **C-index** — ranking quality (the survival AUC). 0.5 random, 1.0 perfect.
* **Integrated Brier Score (IBS)** — accuracy of the predicted *probabilities* (lower = better).
* **Time-dependent AUC** — AUC for "event by time t" as t varies.""")

code(r'''from sksurv.metrics import (concordance_index_censored,
                            integrated_brier_score, cumulative_dynamic_auc)

def evaluate_survival(model, X_te, y_te, e_te, t_te, times):
    """C-index, IBS, and mean time-dependent AUC for any sksurv model with predict_survival_function."""
    risk = model.predict(X_te)
    cidx = concordance_index_censored(e_te.astype(bool), t_te, risk)[0]
    # survival-probability matrix (n_samples x n_times) for the Brier score
    surv_fns = model.predict_survival_function(X_te)
    surv_prob = np.row_stack([fn(times) for fn in surv_fns])
    ibs = integrated_brier_score(y_tr, y_te, surv_prob, times)
    auc_t, auc_mean = cumulative_dynamic_auc(y_tr, y_te, risk, times)
    return {'C-index': round(cidx, 4), 'IBS': round(ibs, 4), 'mean time-AUC': round(auc_mean, 4)}

print('RSF :', evaluate_survival(rsf, X_te, y_te, e_te, t_te, EVAL_TIMES))
print('GBM :', evaluate_survival(gbm, X_te, y_te, e_te, t_te, EVAL_TIMES))
''')

code(r'''# --- K-fold cross-validation harness for a survival model ---
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest

def cv_cindex(make_model, X, t, e, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X):
        m = make_model()
        m.fit(X.iloc[tr], Surv.from_arrays(e[tr].astype(bool), t[tr]))
        risk = m.predict(X.iloc[va])
        scores.append(concordance_index_censored(e[va].astype(bool), t[va], risk)[0])
    return np.mean(scores), np.std(scores)

mean_c, std_c = cv_cindex(lambda: RandomSurvivalForest(n_estimators=100, min_samples_leaf=20,
                                                       n_jobs=-1, random_state=SEED), X, t, e)
print(f'RSF 5-fold C-index: {mean_c:.4f} +/- {std_c:.4f}')
''')

# ============================================================ 9 bridge to classification
md(r"""## §9 — Survival → fixed-horizon classification (your churn case)

If the question is **"will the event happen by a fixed time H?"** (e.g. *churn in April*), you can
collapse the survival problem into a **binary classification** at horizon `H` — which is exactly what
your FictiPay pipeline did. Caveat: only customers observed *past* H (or who had the event before H)
have a known label; others are censored before H and must be dropped or handled with care.""")

code(r'''from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

H = float(np.percentile(df['time'], 50))   # pick a horizon
# label = 1 if event happened by H; known only if event-by-H OR observed beyond H
known = (t_tr <= H) & (e_tr == 1) | (t_tr > H)
y_bin_tr = ((t_tr <= H) & (e_tr == 1)).astype(int)
known_te = (t_te <= H) & (e_te == 1) | (t_te > H)
y_bin_te = ((t_te <= H) & (e_te == 1)).astype(int)

clf = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr[known], y_bin_tr[known])
auc = roc_auc_score(y_bin_te[known_te], clf.predict_proba(X_te[known_te])[:, 1])
print(f'horizon H={H:.2f}  ->  binary "event by H?" AUC = {auc:.4f}  '
      f'(dropped {(~known).mean():.0%} of train as censored-before-H)')
print('Use survival models instead when you need the TIMING or a curve across many horizons.')
''')

# ============================================================ 10 cheat cells
md(r"""## §10 — One-line cheat cells (cram sheet)

```python
# Kaplan–Meier
KaplanMeierFitter().fit(t, e).plot_survival_function();  kmf.median_survival_time_

# Log-rank (2 groups)              -> small p = curves differ
logrank_test(tA, tB, eA, eB).p_value

# Cox + hazard ratios              -> exp(coef) column = HR
CoxPHFitter().fit(df, 'time', 'event').print_summary()
proportional_hazard_test(cph, df).print_summary()        # PH check

# Weibull shape                    -> rho_>1 rises, =1 flat, <1 falls
WeibullFitter().fit(t, e).rho_

# AFT (predict actual time, extrapolate)
WeibullAFTFitter().fit(df, 'time', 'event').predict_median(X)

# Random Survival Forest
RandomSurvivalForest().fit(X, Surv.from_arrays(e.astype(bool), t)).predict(X)

# Metrics
concordance_index_censored(e.astype(bool), t, risk)[0]   # C-index
integrated_brier_score(y_train, y_test, surv_prob, times)
cumulative_dynamic_auc(y_train, y_test, risk, times)[1]  # mean time-AUC

# XGBoost-AFT censoring encoding
dm.set_float_info('label_lower_bound', t)
dm.set_float_info('label_upper_bound', np.where(e==1, t, np.inf))
```
""")

# ============================================================ write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'survival_cookbook.ipynb')
with open(out, 'w') as fh:
    json.dump(nb, fh, indent=1)
print(f'wrote {out} ({len(cells)} cells)')
