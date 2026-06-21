"""Builds notebook.ipynb — FictiPay Round 2 SURVIVAL pipeline (NSUCEC Datathon final).
Run: python3 proto/build_survival_notebook.py
Reuses the Round-1 DuckDB feature engineering verbatim (covariates for the survival model);
replaces target/model/eval/submission with survival-specific stages.
"""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})

# ============================================================ title
md(r"""# FictiPay Churn — Round 2: Customer Lifetime & Survival Analysis
**Bkash NSUCEC Datathon · Final Round · Team Cybernauts**

Round 1 asked *whether* a customer churns. Round 2 asks **when**. We model the full survival
curve and instantaneous hazard from the **Jan–Mar 2024** history.

* **Churn** = 30 consecutive days with no transaction. Survival time **T** is measured from the
  **Mar 31** cutoff; customers still active on **Jun 30** are **right-censored** at T = 91.
* **Submit** (one row per `test.csv` customer): `ACCOUNT_ID, RISK_SCORE, SURV_PROB_30D,
  SURV_PROB_60D, SURV_PROB_90D` with `SURV_PROB_30D ≥ SURV_PROB_60D ≥ SURV_PROB_90D`.
* **Scored:** Concordance index on `RISK_SCORE` (40 pts) + Integrated Brier Score on the
  `SURV_PROB_*` columns (30 pts).

| Stage | Section |
|---|---|
| 1 | Large-scale data handling (DuckDB out-of-core) |
| 2 | Feature engineering — 130+ behavioural / temporal / balance / graph covariates |
| 3 | Feature quality & Pareto handling |
| 4 | The survival target + the 3-horizon reduction + **leakage audit (TrxID is now randomised)** |
| 5 | Survival models — 3-horizon GBM, XGBoost Cox, Random Survival Forest, Cox PH |
| 6 | Hyperparameter tuning (Optuna) |
| 7 | Explainability — SHAP, hazard ratios, survival curves |
| 8 | Business recommendations — time-sensitive retention |
| — | Submission (`predictions.csv`) + OOF c-index / IBS scoring + report/deck |
""")

# ============================================================ setup
code(r"""# ============================== 0. Setup ==============================
import sys, subprocess, importlib, os, glob, time, json, warnings, gc
warnings.filterwarnings('ignore')

def ensure(pkgs):
    missing = []
    for pip_name, mod in pkgs:
        try: importlib.import_module(mod)
        except ImportError: missing.append(pip_name)
    if missing:
        print('installing:', missing)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + missing, check=False)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'duckdb>=1.1'], check=False)
ensure([('duckdb', 'duckdb'), ('lightgbm', 'lightgbm'), ('xgboost', 'xgboost'),
        ('catboost', 'catboost'), ('optuna', 'optuna'), ('shap', 'shap'),
        ('lifelines', 'lifelines'), ('scikit-survival', 'sksurv'), ('psutil', 'psutil')])

import numpy as np, pandas as pd
import duckdb, psutil
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SEED = 42; np.random.seed(SEED)
FAST = os.environ.get('FAST', '0') == '1'   # scaled-down smoke test
HORIZONS = [30, 60, 90]                      # the three scored survival horizons (days from Mar 31)
CENSOR_T = 91                                # right-censoring time (Jun 30)

def find_data_dir():
    # 1) explicit known Kaggle mounts for this competition's dataset (handles both nestings)
    for p in ['/kaggle/input/datasets/mohibulhasantarek/bkash-presents-nsucec-datathon-final/public',
              '/kaggle/input/bkash-presents-nsucec-datathon-final/public',
              '/kaggle/input/bkash-presents-nsucec-datathon-final']:
        if os.path.exists(os.path.join(p, 'kyc.parquet')):
            return p
    # 2) recursive search; ignore any Round-1 'prev' copy, prefer a folder literally named 'public'
    hits = [h for h in glob.glob('/kaggle/input/**/kyc.parquet', recursive=True) if 'prev' not in h.lower()]
    pub = [h for h in hits if os.path.basename(os.path.dirname(h)) == 'public']
    if pub:  return os.path.dirname(pub[0])
    if hits: return os.path.dirname(hits[0])
    # 3) local fallback (smoke tests)
    for c in ['public', '../public', 'data', '.']:
        if os.path.exists(os.path.join(c, 'kyc.parquet')): return os.path.abspath(c)
    raise FileNotFoundError('Could not locate kyc.parquet — check dataset attachment')

DATA = find_data_dir()
def find_glob(sub, pat):
    g = os.path.join(DATA, sub, pat)
    if glob.glob(g): return g
    hits = glob.glob(os.path.join(DATA, '**', pat), recursive=True)
    assert hits, f'no files matching {pat} under {DATA}'
    return os.path.join(os.path.dirname(hits[0]), pat)

TRX_GLOB = find_glob('transactions', 'trx_*.parquet')
BAL_GLOB = find_glob('dayend_balance', 'balance_*.parquet')
for d in ['explainability', 'figures', 'tmp_duckdb']:
    os.makedirs(d, exist_ok=True)

CUTOFF = pd.Timestamp('2024-04-01')   # features computed as-of this instant (T measured from Mar 31)
OBS_START, OBS_END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-31')
N_THREADS = os.cpu_count(); FE_THREADS = min(N_THREADS, 4)
RAM_GB = psutil.virtual_memory().total / 1e9
import importlib.util
HAS = lambda m: importlib.util.find_spec(m) is not None
HAS_CAT, HAS_SKSURV, HAS_LIFE = HAS('catboost'), HAS('sksurv'), HAS('lifelines')
HAS_SHAP, HAS_OPTUNA = HAS('shap'), HAS('optuna')
print('DATA:', DATA)
print(f'duckdb {duckdb.__version__} | cpus={N_THREADS} ram={RAM_GB:.1f}GB FAST={FAST}')
print(f'optional packages: catboost={HAS_CAT} sksurv={HAS_SKSURV} lifelines={HAS_LIFE} shap={HAS_SHAP} optuna={HAS_OPTUNA}')
""")

# ============================================================ helpers (offline-safe)
md(r"""### Helpers & metrics — *offline-safe*

All scoring (c-index, Integrated Brier Score) is implemented in **pure NumPy** so the notebook runs
**without Internet** even if `scikit-survival` is absent (it isn't in Kaggle's default image). If
`scikit-survival` *is* present, the exact concordance is used; otherwise an accurate subsampled
estimate. `lifelines`/`shap`/`optuna`/`catboost` are all optional and degrade gracefully.""")

code(r"""# ============================== Helpers & metrics (no hard dependency on scikit-survival) ==============================
from scipy.stats import rankdata as _rankdata
GRID = [30, 60, 90]                                   # horizons for the expected-survival risk (== submit horizons)
SUBMIT_IDX = [GRID.index(h) for h in HORIZONS]        # positions of 30/60/90 within GRID

def c_index(dur, evt, risk, exact=False):
    # Harrell's concordance (right-censored). Default = fast subsampled NumPy estimate (~1e-3
    # accurate, offline-safe, runs in ~1s). exact=True uses scikit-survival (slow on 595k rows).
    if exact and HAS_SKSURV:
        from sksurv.metrics import concordance_index_censored
        return float(concordance_index_censored(evt.astype(bool), dur, risk)[0])
    rng = np.random.default_rng(0); ev = np.where(evt == 1)[0]
    if len(ev) == 0: return 0.5
    i = rng.choice(ev, 3_000_000); j = rng.integers(0, len(dur), 3_000_000)
    ok = dur[j] > dur[i]; i, j = i[ok], j[ok]            # comparable: i (event) fails before j
    if len(i) == 0: return 0.5
    ri, rj = risk[i], risk[j]
    return float(((ri > rj).sum() + 0.5 * (ri == rj).sum()) / len(i))

def brier_at_horizons(dur, S):
    return {h: float(np.mean(((dur > h).astype(float) - S[:, i]) ** 2)) for i, h in enumerate(HORIZONS)}

def ibs_score(dur_tr, evt_tr, dur, evt, S):
    # Integrated Brier Score over [30,90] by trapezoid. Every horizon <= the single censoring
    # time (91), so the Brier labels are fully observed and this equals the IPCW IBS here.
    b = np.array([np.mean(((dur > h).astype(float) - S[:, i]) ** 2) for i, h in enumerate(HORIZONS)])
    xs = np.array(HORIZONS, float)
    return float(((b[:-1] + b[1:]) * 0.5 * np.diff(xs)).sum() / (xs[-1] - xs[0]))

def comp_score(ci, ibs):
    return max(0, (ci - 0.5) / 0.5) * 40 + max(0, (0.05 - ibs) / 0.05) * 30

def exp_surv(Sg):                                       # integral of S(t) over GRID (NumPy 1.x & 2.x safe)
    xs = np.array([0] + GRID, float)
    S = np.concatenate([np.ones((Sg.shape[0], 1)), Sg], axis=1)
    return ((S[:, :-1] + S[:, 1:]) * 0.5 * np.diff(xs)).sum(axis=1)

def risk_set(exp_, cox_, Sg_, S_):                      # candidate RISK_SCOREs (auto-picked by OOF c-index)
    return {'1 - S90':          1 - S_[:, 2],
            'expected_surv(-)': -exp_,
            'sum(1-S_grid)':    (1 - Sg_).sum(axis=1),
            'xgb_cox':          cox_,
            'cox+exp blend':    0.5 * _rankdata(cox_) + 0.5 * _rankdata(-exp_)}

def enforce_monotone(S):                                # SURV_PROB_30D >= 60D >= 90D, clipped to [0,1]
    return np.minimum.accumulate(np.clip(S, 0, 1), axis=1)
""")

# ============================================================ preflight
md(r"""### Preflight — *fail fast (≈20 s) before the ~40-min run*

This validates packages, locates every data file, and runs the **entire modelling + scoring +
submission code path on tiny synthetic data** — so version/API bugs (e.g. a removed NumPy function)
surface in seconds, not after a 40-minute cross-validation. Set `PREFLIGHT = False` to skip.""")

code(r"""# ============================== PREFLIGHT — catch problems in ~20s, before the long run ==============================
PREFLIGHT = True
if PREFLIGHT:
    print('[1/3] package check')
    req = ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'xgboost', 'duckdb', 'matplotlib']
    miss_req = [m for m in req if not HAS(m)]
    if miss_req:
        raise RuntimeError(f'MISSING REQUIRED packages {miss_req} — enable Internet (Settings) or '
                           f'attach a wheels dataset; the pipeline cannot run without these.')
    print('   required OK :', ', '.join(req))
    print('   optional    : catboost={} sksurv={} lifelines={} shap={} optuna={}'.format(
          HAS_CAT, HAS_SKSURV, HAS_LIFE, HAS_SHAP, HAS_OPTUNA),
          '(missing ones are skipped gracefully)')

    print('[2/3] data files check')
    for p in [f'{DATA}/kyc.parquet', f'{DATA}/train_labels.csv', f'{DATA}/test.csv']:
        assert os.path.exists(p), f'MISSING data file: {p}'
    assert glob.glob(TRX_GLOB), f'no transactions at {TRX_GLOB}'
    assert glob.glob(BAL_GLOB), f'no balances at {BAL_GLOB}'
    print('   data OK at', DATA)

    print('[3/3] end-to-end CODE self-test on synthetic data (exercises every downstream op)')
    import lightgbm as _lgb, xgboost as _xgb
    from sklearn.isotonic import IsotonicRegression as _ISO
    _r = np.random.default_rng(0); _n = 600
    _X = pd.DataFrame(_r.normal(size=(_n, 8)), columns=[f'c{i}' for i in range(8)])
    _dur = _r.integers(0, 92, _n).astype(float); _evt = (_dur < 91).astype(int)
    _Sg = np.zeros((_n, len(GRID)))
    for _gi, _h in enumerate(GRID):
        _yb = (_dur > _h).astype(int)
        if _yb.sum() in (0, _n): _Sg[:, _gi] = _yb.mean(); continue
        _Sg[:, _gi] = _lgb.LGBMClassifier(n_estimators=25, verbose=-1).fit(_X, _yb).predict_proba(_X)[:, 1]
    _Sx = np.column_stack([_xgb.XGBClassifier(n_estimators=25, verbosity=0).fit(
                           _X, (_dur > h).astype(int)).predict_proba(_X)[:, 1] for h in HORIZONS])
    _parts = [_Sg[:, SUBMIT_IDX], _Sx]
    if HAS_CAT:
        from catboost import CatBoostClassifier as _CB
        _parts.append(np.column_stack([_CB(iterations=25, verbose=0).fit(
                      _X, (_dur > h).astype(int)).predict_proba(_X)[:, 1] for h in HORIZONS]))
    _S = enforce_monotone(np.mean(_parts, axis=0))
    _cand = risk_set(exp_surv(_Sg), _r.normal(size=_n), _Sg, _S)
    _best = max(_cand, key=lambda k: c_index(_dur, _evt, _cand[k]))
    for _i, _h in enumerate(HORIZONS):
        _ISO(out_of_bounds='clip', y_min=0, y_max=1).fit(_S[:, _i], (_dur > _h).astype(float)).transform(_S[:, _i])
    _sub = pd.DataFrame({'ACCOUNT_ID': [f'X{i}' for i in range(_n)], 'RISK_SCORE': _cand[_best],
                         'SURV_PROB_30D': _S[:, 0], 'SURV_PROB_60D': _S[:, 1], 'SURV_PROB_90D': _S[:, 2]})
    assert (_sub.SURV_PROB_30D >= _sub.SURV_PROB_60D - 1e-9).all() and (_sub.SURV_PROB_60D >= _sub.SURV_PROB_90D - 1e-9).all()
    _ci, _ibs = c_index(_dur, _evt, _cand[_best]), ibs_score(_dur, _evt, _dur, _evt, _S)
    print(f'   self-test OK | synthetic c-index={_ci:.3f} ibs={_ibs:.3f} score={comp_score(_ci,_ibs):.1f} | risk={_best}')
    _eta = '~20 min' if not HAS_CAT else '~40 min'
    print(f'PREFLIGHT PASSED ✓ — full run is safe (feature build ~3 min, CV {_eta}). CatBoost ensemble: {HAS_CAT}.')
    del _X, _Sg, _Sx, _S, _parts
""")

# ============================================================ stage 1
md(r"""## Stage 1 — Large-Scale Data Handling

**DuckDB** — a vectorised, multi-threaded, **out-of-core** OLAP engine that streams aggregations
over Parquet far larger than RAM and spills to disk automatically. We chose it over Spark (JVM +
shuffle overhead, only pays off across a cluster) and Dask (slower Python task graph for wide
groupbys); on single-node aggregation benchmarks (TPC-H, db-benchmark) DuckDB beats both.

* **Column pruning + predicate pushdown** on the month-partitioned Parquet files; only the final
  per-customer feature matrix (~595 K × ~140) enters pandas.
* **Sampling** for EDA via `USING SAMPLE` (reservoir). A 100 K sample gives ±0.1 pp standard error
  on the ~5 % event rate — ample for distribution decisions.
* **Leakage guard:** assert every record lies in the Jan–Mar observation window before building
  features (Stage 4 also audits the TrxID structure for the Round-1 leak).
""")

code(r"""# ============================== Stage 1: ingestion ==============================
con = duckdb.connect()
con.execute(f"PRAGMA threads={FE_THREADS}")
con.execute(f"PRAGMA memory_limit='{max(2, int(RAM_GB * 0.55))}GB'")
con.execute("PRAGMA temp_directory='tmp_duckdb'")
con.execute("SET preserve_insertion_order=false")

con.execute(f'''CREATE OR REPLACE VIEW trx AS
SELECT TrxID, SRC_ACCOUNT, DST_ACCOUNT, TRX_TYPE, TRX_AMT,
       CAST(substr(TRX_DATETIME, 1, 10) AS DATE) AS d,
       CAST(substr(TRX_DATETIME, 12, 2) AS INT)  AS hr
FROM read_parquet('{TRX_GLOB}');''')
con.execute(f'''CREATE OR REPLACE VIEW bal AS
SELECT ACCOUNT_ID, CAST(DATE AS DATE) AS d, AVAILABLE_BALANCE AS b
FROM read_parquet('{BAL_GLOB}');''')

t0 = time.time()
sizes = con.execute(f'''SELECT 'transactions' AS tbl, count(*) AS nrows FROM trx
UNION ALL SELECT 'dayend_balance', count(*) FROM bal
UNION ALL SELECT 'kyc', count(*) FROM read_parquet('{DATA}/kyc.parquet')''').df()
print(sizes.to_string(index=False), f'\n(counted in {time.time()-t0:.1f}s — metadata only)')

rng = con.execute("SELECT min(d), max(d) FROM trx").fetchone()
brng = con.execute("SELECT min(d), max(d) FROM bal").fetchone()
print('trx dates:', rng, '| bal dates:', brng)
assert rng[1] <= OBS_END.date() and brng[1] <= OBS_END.date(), 'LEAKAGE: data beyond observation window!'
print('leakage guard passed: all records <=', OBS_END.date())

# survival labels (Round 2): DURATION_DAYS (0-91), EVENT_FLAG (1=churn observed, 0=censored at 91)
labels  = pd.read_csv(f'{DATA}/train_labels.csv')
test_ids = pd.read_csv(f'{DATA}/test.csv')
print(f'\ntrain={len(labels):,}  test={len(test_ids):,}')
print('event rate =', round(labels.EVENT_FLAG.mean(), 4),
      '| censored (EVENT=0) always DURATION=91:', bool((labels.loc[labels.EVENT_FLAG==0,'DURATION_DAYS']==91).all()))
""")

# ============================================================ stage 2 (feature build — reused)
md(r"""## Stage 2 — Feature Engineering (survival covariates)

The same rich behavioural feature pipeline we built in Round 1 is exactly what a survival model
needs as covariates: **recency, personal rhythm, frequency, balance dynamics, weekly trajectory,
network/peer activity**. All are computed **only** from the Jan–Mar window (cutoff = Apr 1).
Six families — full annotated list written to `features.md`.

1. **Outgoing activity** — nested-window counts/amounts (3/7/14/30/60/90 d), per-type counts &
   recency, active days/weeks, monthly counts, time-of-day mix.
2. **Personal rhythm (survival-style)** — inter-transaction gap stats, `gap_z` = *(silence −
   personal mean gap)/personal std*, `recency_over_maxgap`, silent-window share, Poisson silence prob.
3. **Trajectory** — 13 weekly counts + slope / trailing-silent-weeks (the *shape* of decay).
4. **Incoming & network** — P2P received, recurring counterparties, peer-graph liveness.
5. **Balance dynamics** — level, volatility, slope, zero-balance days, days since last change, drain ratio.
6. **Profile (KYC)** — tenure (open date may pre-date 2024), region, gender.
""")

code(r"""# ============================== Stage 2a: heavy aggregation in DuckDB (wave 1) ==============================
t0 = time.time()
con.execute('''CREATE OR REPLACE TABLE daily AS
SELECT SRC_ACCOUNT AS id, d, count(*) AS c FROM trx GROUP BY 1, 2;''')
print(f'daily          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_cnt AS
SELECT id AS ACCOUNT_ID,
  sum(c) AS cnt_90d,
  sum(c) FILTER (d >= DATE '2024-02-01')                           AS cnt_60d,
  sum(c) FILTER (d >= DATE '2024-03-02')                           AS cnt_30d,
  sum(c) FILTER (d >= DATE '2024-03-18')                           AS cnt_14d,
  sum(c) FILTER (d >= DATE '2024-03-25')                           AS cnt_7d,
  sum(c) FILTER (d >= DATE '2024-03-29')                           AS cnt_3d,
  sum(c) FILTER (d <  DATE '2024-02-01')                           AS cnt_jan,
  sum(c) FILTER (d >= DATE '2024-02-01' AND d < DATE '2024-03-01') AS cnt_feb,
  sum(c) FILTER (d >= DATE '2024-03-01')                           AS cnt_mar,
  max(d) AS last_d, min(d) AS first_d,
  count(*) AS active_days,
  count(*) FILTER (d >= DATE '2024-03-02')                         AS active_days_30d,
  count(DISTINCT date_trunc('week', d))                            AS active_weeks,
  count(DISTINCT date_trunc('month', d))                           AS n_active_months
FROM daily GROUP BY 1;''')
print(f'f_cnt          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_amt AS
SELECT SRC_ACCOUNT AS ACCOUNT_ID,
  min(date_diff('hour', CAST(d AS TIMESTAMP) + to_hours(hr), TIMESTAMP '2024-04-01 00:00:00')) AS hours_since_last,
  sum(TRX_AMT) AS amt_sum_90d, avg(TRX_AMT) AS amt_mean_90d, stddev(TRX_AMT) AS amt_std_90d, max(TRX_AMT) AS amt_max_90d,
  sum(TRX_AMT) FILTER (d >= DATE '2024-03-02') AS amt_sum_30d, avg(TRX_AMT) FILTER (d >= DATE '2024-03-02') AS amt_mean_30d,
  count(*) FILTER (TRX_TYPE = 'P2P') AS cnt_p2p, count(*) FILTER (TRX_TYPE = 'MerchantPay') AS cnt_merchant,
  count(*) FILTER (TRX_TYPE = 'BillPay') AS cnt_bill, count(*) FILTER (TRX_TYPE = 'CashIn') AS cnt_cashin,
  count(*) FILTER (TRX_TYPE = 'CashOut') AS cnt_cashout,
  count(*) FILTER (TRX_TYPE = 'P2P' AND d >= DATE '2024-03-02') AS cnt_p2p_30d,
  count(*) FILTER (TRX_TYPE = 'MerchantPay' AND d >= DATE '2024-03-02') AS cnt_merchant_30d,
  count(*) FILTER (TRX_TYPE = 'BillPay' AND d >= DATE '2024-03-02') AS cnt_bill_30d,
  count(*) FILTER (TRX_TYPE = 'CashIn' AND d >= DATE '2024-03-02') AS cnt_cashin_30d,
  count(*) FILTER (TRX_TYPE = 'CashOut' AND d >= DATE '2024-03-02') AS cnt_cashout_30d,
  date_diff('day', max(d) FILTER (TRX_TYPE='P2P'),         DATE '2024-04-01') AS rec_p2p,
  date_diff('day', max(d) FILTER (TRX_TYPE='MerchantPay'), DATE '2024-04-01') AS rec_merchant,
  date_diff('day', max(d) FILTER (TRX_TYPE='BillPay'),     DATE '2024-04-01') AS rec_bill,
  date_diff('day', max(d) FILTER (TRX_TYPE='CashIn'),      DATE '2024-04-01') AS rec_cashin,
  date_diff('day', max(d) FILTER (TRX_TYPE='CashOut'),     DATE '2024-04-01') AS rec_cashout,
  sum(TRX_AMT) FILTER (TRX_TYPE = 'CashIn') AS amt_cashin, sum(TRX_AMT) FILTER (TRX_TYPE = 'CashOut') AS amt_cashout,
  count(*) FILTER (hr >= 18) AS cnt_evening, count(*) FILTER (dayofweek(d) IN (0, 6)) AS cnt_weekend
FROM trx GROUP BY 1;''')
print(f'f_amt          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_gap AS
WITH g AS (SELECT id, date_diff('day', lag(d) OVER (PARTITION BY id ORDER BY d), d) AS gap FROM daily)
SELECT id AS ACCOUNT_ID, max(gap) AS gap_max, avg(gap) AS gap_mean, stddev(gap) AS gap_std,
  sum(greatest(0, gap - 30)) AS dead30_inner,
  count(*) FILTER (gap >= 7)  AS n_gap7,    -- # of silent runs >= 7 / 14 / 21 days
  count(*) FILTER (gap >= 14) AS n_gap14,
  count(*) FILTER (gap >= 21) AS n_gap21,
  quantile_cont(gap, 0.9)     AS gap_p90    -- this customer's 90th-percentile inter-txn gap
FROM g WHERE gap IS NOT NULL GROUP BY 1;''')
print(f'f_gap          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_in AS
SELECT DST_ACCOUNT AS ACCOUNT_ID, count(*) AS in_cnt_90d,
  count(*) FILTER (d >= DATE '2024-03-02') AS in_cnt_30d, sum(TRX_AMT) AS in_amt_90d,
  max(d) AS in_last_d, approx_count_distinct(SRC_ACCOUNT) AS in_n_senders
FROM trx WHERE TRX_TYPE = 'P2P' AND SRC_ACCOUNT <> DST_ACCOUNT GROUP BY 1;''')
print(f'f_in           {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_bal AS
SELECT ACCOUNT_ID, arg_max(b, d) AS bal_last, avg(b) AS bal_mean_90d, stddev(b) AS bal_std_90d,
  min(b) AS bal_min_90d, max(b) AS bal_max_90d,
  avg(b) FILTER (d >= DATE '2024-03-02') AS bal_mean_30d, stddev(b) FILTER (d >= DATE '2024-03-02') AS bal_std_30d,
  avg(b) FILTER (d <  DATE '2024-02-01') AS bal_mean_jan, avg(b) FILTER (d >= DATE '2024-03-25') AS bal_mean_7d,
  count(*) FILTER (b <= 1.0) AS zero_bal_days_90d,
  count(*) FILTER (b <= 1.0 AND d >= DATE '2024-03-02') AS zero_bal_days_30d,
  count(*) FILTER (b <= 1.0 AND d >= DATE '2024-03-25') AS zero_bal_days_7d,
  regr_slope(b, date_diff('day', DATE '2024-01-01', d)) AS bal_slope
FROM bal GROUP BY 1;''')
print(f'f_bal          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_balchg AS
WITH x AS (SELECT ACCOUNT_ID AS id, d, b - lag(b) OVER (PARTITION BY ACCOUNT_ID ORDER BY d) AS diff FROM bal)
SELECT id AS ACCOUNT_ID, max(d) FILTER (diff IS NOT NULL AND diff <> 0) AS bal_last_chg_d,
  count(*) FILTER (diff <> 0 AND d >= DATE '2024-03-02') AS bal_chg_days_30d,
  count(*) FILTER (diff <> 0) AS bal_chg_days_90d, stddev(diff) AS bal_diff_std,
  avg(abs(diff)) AS bal_diff_absmean, sum(CASE WHEN diff < 0 THEN -diff ELSE 0 END) AS bal_outflow_90d
FROM x GROUP BY 1;''')
print(f'f_balchg       {time.time()-t0:6.0f}s')
""")

code(r"""# ============================== Stage 2a: wave 2 — trajectory, recurrence, peer graph ==============================
wk_aggs = ",\n  ".join(f"sum(c) FILTER (wk = {i}) AS wk{i}_cnt" for i in range(13))
con.execute(f'''CREATE OR REPLACE TABLE f_wk AS
WITH tw AS (SELECT id, c, CAST(floor(date_diff('day', DATE '2024-01-01', d) / 7) AS INT) AS wk FROM daily)
SELECT id AS ACCOUNT_ID, {wk_aggs} FROM tw GROUP BY 1;''')
print(f'f_wk           {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE pair_months AS
WITH ptm AS (SELECT SRC_ACCOUNT s, DST_ACCOUNT t, substr(DST_ACCOUNT, 1, 4) pre, date_trunc('month', d) mo
             FROM trx GROUP BY 1, 2, 3, 4)
SELECT s, t, pre, count(*) AS m FROM ptm GROUP BY 1, 2, 3;''')
con.execute('''CREATE OR REPLACE TABLE f_net AS
SELECT s AS ACCOUNT_ID, count(*) AS n_counterparties, count(*) FILTER (pre = 'MRCH') AS n_merchants,
  count(*) FILTER (pre = 'CUST' AND s <> t) AS n_p2p_peers, count(*) FILTER (m = 3) AS n_recur3_all,
  count(*) FILTER (m >= 2) AS n_recur2_all, count(*) FILTER (m = 3 AND pre = 'BILL') AS n_recur3_bill,
  count(*) FILTER (m = 3 AND pre = 'MRCH') AS n_recur3_mrch, count(*) FILTER (m = 3 AND pre = 'CUST' AND s <> t) AS n_recur3_p2p
FROM pair_months GROUP BY 1;''')
print(f'f_net          {time.time()-t0:6.0f}s')

con.execute('''CREATE OR REPLACE TABLE f_peer AS
WITH e AS (SELECT s AS a, t AS p FROM pair_months WHERE pre = 'CUST' AND s <> t
           UNION SELECT t AS a, s AS p FROM pair_months WHERE pre = 'CUST' AND s <> t)
SELECT a AS ACCOUNT_ID, count(*) AS n_peers_2way, avg(f.cnt_30d) AS peer_avg_cnt30,
  avg(date_diff('day', f.last_d, DATE '2024-04-01')) AS peer_avg_recency,
  min(date_diff('day', f.last_d, DATE '2024-04-01')) AS peer_min_recency, max(f.cnt_30d) AS peer_max_cnt30
FROM e JOIN f_cnt f ON f.ACCOUNT_ID = e.p GROUP BY 1;''')
con.execute('DROP TABLE pair_months')
print(f'f_peer         {time.time()-t0:6.0f}s')

wkb = ",\n  ".join(f"avg(b) FILTER (CAST(floor(date_diff('day', DATE '2024-01-01', d) / 7) AS INT) = {i}) AS wk{i}_bal"
                   for i in [0, 4, 8, 10, 11, 12])
con.execute(f'''CREATE OR REPLACE TABLE f_wkb AS SELECT ACCOUNT_ID, {wkb} FROM bal GROUP BY 1;''')
print(f'f_wkb          {time.time()-t0:6.0f}s')

feat = con.execute(f'''
SELECT k.ACCOUNT_ID, date_diff('day', CAST(k.ACCOUNT_OPEN_DATE AS DATE), DATE '2024-04-01') AS tenure_days,
       k.GENDER, k.REGION,
       o.* EXCLUDE (ACCOUNT_ID), a.* EXCLUDE (ACCOUNT_ID), g.* EXCLUDE (ACCOUNT_ID), i.* EXCLUDE (ACCOUNT_ID),
       b.* EXCLUDE (ACCOUNT_ID), c.* EXCLUDE (ACCOUNT_ID), w.* EXCLUDE (ACCOUNT_ID),
       r.* EXCLUDE (ACCOUNT_ID), p.* EXCLUDE (ACCOUNT_ID), wb.* EXCLUDE (ACCOUNT_ID)
FROM read_parquet('{DATA}/kyc.parquet') k
LEFT JOIN f_cnt o USING (ACCOUNT_ID)
LEFT JOIN f_amt a ON a.ACCOUNT_ID=k.ACCOUNT_ID LEFT JOIN f_gap g ON g.ACCOUNT_ID=k.ACCOUNT_ID
LEFT JOIN f_in i ON i.ACCOUNT_ID=k.ACCOUNT_ID LEFT JOIN f_bal b ON b.ACCOUNT_ID=k.ACCOUNT_ID
LEFT JOIN f_balchg c ON c.ACCOUNT_ID=k.ACCOUNT_ID LEFT JOIN f_wk w ON w.ACCOUNT_ID=k.ACCOUNT_ID
LEFT JOIN f_net r ON r.ACCOUNT_ID=k.ACCOUNT_ID LEFT JOIN f_peer p ON p.ACCOUNT_ID=k.ACCOUNT_ID
LEFT JOIN f_wkb wb ON wb.ACCOUNT_ID=k.ACCOUNT_ID
WHERE k.ACCOUNT_TYPE = 'Customer' ''').df()
for tbl in ['daily','f_cnt','f_amt','f_gap','f_in','f_bal','f_balchg','f_wk','f_net','f_peer','f_wkb']:
    con.execute(f'DROP TABLE {tbl}')
print(f'feature table  {time.time()-t0:6.0f}s  shape={feat.shape}')
""")

code(r"""# ============================== Stage 2b: derived features (pandas) ==============================
eps = 1e-6; f = feat
for c in ['last_d', 'first_d', 'in_last_d', 'bal_last_chg_d']:
    f[c] = pd.to_datetime(f[c])
f['recency_days']        = (CUTOFF - f['last_d']).dt.days.fillna(120)
f['first_seen_days']     = (CUTOFF - f['first_d']).dt.days.fillna(120)
f['in_recency_days']     = (CUTOFF - f['in_last_d']).dt.days.fillna(120)
f['bal_stagnation_days'] = (CUTOFF - f['bal_last_chg_d']).dt.days.fillna(120)
f['money_recency']       = f[['recency_days', 'bal_stagnation_days']].min(axis=1)
f.drop(columns=['last_d', 'first_d', 'in_last_d', 'bal_last_chg_d'], inplace=True)
f['hours_since_last'] = f['hours_since_last'].fillna(120 * 24)
for c in ['rec_p2p','rec_merchant','rec_bill','rec_cashin','rec_cashout']: f[c] = f[c].fillna(120)
f['gap_mean'] = f['gap_mean'].fillna(120); f['gap_max'] = f['gap_max'].fillna(120); f['gap_std'] = f['gap_std'].fillna(0)
f['gap_z'] = (f['recency_days'] - f['gap_mean']) / (f['gap_std'] + 1)
f['recency_over_maxgap'] = f['recency_days'] / (f['gap_max'] + 1)
f['recency_over_meangap'] = f['recency_days'] / (f['gap_mean'] + 1)
lead_empty = (91 - f['first_seen_days']).clip(lower=0)
f['dead30_share'] = (f['dead30_inner'].fillna(0) + (lead_empty - 29).clip(lower=0)) / 62.0
f['p_silent30_poisson'] = np.exp(-f['cnt_30d'].fillna(0))
wk_cols = [f'wk{i}_cnt' for i in range(13)]; W = f[wk_cols].fillna(0).values
f['trailing_zero_weeks'] = (np.cumprod(W[:, ::-1] == 0, axis=1)).sum(1)
f['wk_slope'] = (W * (np.arange(13) - 6)).sum(1) / (W.sum(1) + 1)
f['wk_last_over_mean'] = W[:, 12] / (W.mean(1) + eps)
f['wk_last4_over_first4'] = W[:, 9:].sum(1) / (W[:, :4].sum(1) + 1)
f['trend_30_90'] = f['cnt_30d'] / (f['cnt_90d'] / 3 + eps)
f['trend_7_30']  = f['cnt_7d'] / (f['cnt_30d'] * 7 / 30 + eps)
f['mar_over_janfeb'] = f['cnt_mar'] / ((f['cnt_jan'] + f['cnt_feb']) / 2 + 1)
f['intensity'] = f['cnt_90d'] / (f['active_days'] + eps)
f['active_rate_30d'] = f['active_days_30d'] / 30.0
for t in ['p2p','merchant','bill','cashin','cashout']: f[f'share_{t}'] = f[f'cnt_{t}'] / (f['cnt_90d'] + eps)
f['evening_share'] = f['cnt_evening'] / (f['cnt_90d'] + eps); f['weekend_share'] = f['cnt_weekend'] / (f['cnt_90d'] + eps)
f['in_out_ratio'] = f['in_cnt_90d'] / (f['cnt_90d'] + 1); f['amt_cv'] = f['amt_std_90d'] / (f['amt_mean_90d'] + eps)
f['net_cashflow'] = f['amt_cashin'].fillna(0) - f['amt_cashout'].fillna(0)
f['bal_last_over_mean'] = f['bal_last'] / (f['bal_mean_90d'] + 1)
f['bal_last_over_max'] = f['bal_last'] / (f['bal_max_90d'] + 1)
f['bal_30_over_jan'] = f['bal_mean_30d'] / (f['bal_mean_jan'] + 1)
f['bal_chg_rate_30d'] = f['bal_chg_days_30d'] / 30.0
f['bal_wk12_over_wk0'] = f['wk12_bal'] / (f['wk0_bal'] + 1)
# --- Tier-1 churn-mechanism & velocity features (label = 30 consecutive silent days) ---
for c in ['n_gap7', 'n_gap14', 'n_gap21']: f[c] = f[c].fillna(0)
f['gap_p90'] = f['gap_p90'].fillna(120)
f['max_silent_streak']   = f[['gap_max', 'recency_days']].max(axis=1)         # longest silence incl. trailing run
f['near_churn_gap']      = f['gap_max'] / 30.0                                # worst gap vs the 30-day churn threshold
f['ever_21d_gap']        = (f['gap_max'] >= 21).astype('int8')                # has flirted with the churn rule before
f['silence_over_p90gap'] = f['recency_days'] / (f['gap_p90'] + 1)             # current silence vs personal 90th-pct gap
f['trend_accel']         = f['trend_7_30'] - f['trend_30_90']                 # is the activity decline accelerating?
f['days_to_zero_bal']    = f['bal_last'] / (f['bal_outflow_90d'].fillna(0) / 90 + 1)  # est. days until wallet empties
# --- regularity / burstiness / entropy: how CONSISTENT a customer is, not how active ---
_Wp = W / (W.sum(1, keepdims=True) + eps)
f['wk_entropy']       = -(np.where(_Wp > 0, _Wp * np.log(_Wp + eps), 0)).sum(1)   # activity spread across the 13 weeks
f['wk_cv']            = W.std(1) / (W.mean(1) + eps)                              # burstiness of weekly activity
f['active_week_frac'] = (W > 0).sum(1) / 13.0                                     # fraction of weeks with any activity
f['gap_cv']           = f['gap_std'] / (f['gap_mean'] + 1)                        # irregularity of inter-txn timing
_sh = np.vstack([f[f'share_{t}'].values for t in ['p2p', 'merchant', 'bill', 'cashin', 'cashout']]).T
f['type_entropy']     = -(np.where(_sh > 0, _sh * np.log(_sh + eps), 0)).sum(1)   # diversity of transaction types
for c in ['n_peers_2way','peer_avg_cnt30','peer_max_cnt30','n_recur3_all','n_recur2_all','n_recur3_bill','n_recur3_mrch','n_recur3_p2p']: f[c] = f[c].fillna(0)
for c in ['peer_avg_recency','peer_min_recency']: f[c] = f[c].fillna(120)
f['is_inactive_30d'] = (f['cnt_30d'].fillna(0) == 0).astype('int8')
f['is_inactive_90d'] = (f['cnt_90d'].fillna(0) == 0).astype('int8')
f['no_inflow_90d'] = (f['in_cnt_90d'].fillna(0) == 0).astype('int8')
f['is_drained'] = (f['bal_last'].fillna(0) <= 1.0).astype('int8')
for c in ['GENDER','REGION']: f[c] = f[c].astype('category').cat.codes.astype('int16')
num_cols = f.columns.difference(['ACCOUNT_ID']); f[num_cols] = f[num_cols].astype('float32').fillna(0)

FEATURES = [c for c in f.columns if c != 'ACCOUNT_ID']
train = labels.merge(f, on='ACCOUNT_ID', how='left')
test  = test_ids.merge(f, on='ACCOUNT_ID', how='left')
assert train[FEATURES].notna().all().all() and test[FEATURES].notna().all().all()
del feat, W; gc.collect()
print(f'{len(FEATURES)} features | train {train.shape} | test {test.shape}')
""")

code(r'''# ============================== Stage 2c: features.md ==============================
FEATURE_DOC = {
 'recency_days': 'Days since last transaction — the most direct disengagement signal; the dominant survival covariate.',
 'gap_z': "Z-score of current silence vs the customer's own gap distribution — a 10-day pause alarms a daily user, not a monthly one.",
 'recency_over_maxgap': 'Silence vs longest historical pause — >1 means uncharted dormancy.',
 'bal_stagnation_days': 'Days since the day-end balance last moved — inactivity incl. inflows.',
 'p_silent30_poisson': 'Poisson P(0 events in 30d) given recent rate — analytic hazard proxy.',
 'cnt_3d..cnt_90d': 'Counts over nested windows — level + decay of engagement.',
 'wk0_cnt..wk12_cnt + wk_slope / trailing_zero_weeks': '13-week trajectory & shape — *when* engagement faded.',
 'trend_30_90 / trend_7_30': 'Recent vs longer-window rate — momentum; declining users churn sooner.',
 'active_days / active_weeks': 'Distinct active periods — habit regularity.',
 'rec_p2p..rec_cashout': 'Per-type recency — a habitual bill-payer who stopped is a red flag.',
 'amt_sum_90d / amt_cv': 'Spend level & dispersion — high-value users have higher switching costs.',
 'share_p2p..share_cashout': 'Type mix — bill-payers sticky; cash-out-heavy may be exiting.',
 'n_counterparties / n_recur3_*': 'Network breadth & habitual relationships — lock-in.',
 'peer_avg_recency / peer_avg_cnt30': 'Liveness of P2P partners — churn is socially contagious.',
 'in_cnt_90d / in_recency_days': 'Incoming P2P — receiving money keeps a wallet alive.',
 'bal_last / bal_slope / bal_last_over_max': 'Balance level & trajectory & drain ratio — a draining wallet precedes exit.',
 'tenure_days': 'Account age (open date may pre-date 2024) — loyalty accumulates.',
 'GENDER / REGION': 'Demographics — regional agent density affects usage.',
 'is_inactive_30d / is_drained / no_inflow_90d': 'Zero-inflation indicator flags (Stage 3).',
}
lines = ['# FictiPay Round 2 — Survival Covariate Documentation', '',
         f'Cutoff: {CUTOFF.date()} | Observation: {OBS_START.date()} -> {OBS_END.date()} | T measured from 2024-03-31', '',
         f'Total covariates: **{len(FEATURES)}**. Every column belongs to a documented hypothesis family below.', '',
         '| Feature family | Behavioural hypothesis (why it predicts time-to-churn) |', '|---|---|']
for k, v in FEATURE_DOC.items(): lines.append(f'| `{k}` | {v} |')
open('features.md', 'w').write('\n'.join(lines) + '\n')
print(f'features.md written — {len(FEATURE_DOC)} families, {len(FEATURES)} columns')
''')

# ============================================================ stage 3
md(r"""## Stage 3 — Feature Quality & Pareto Analysis

Fintech behaviour is textbook **Pareto** (few power-users dominate volume) and **zero-inflated**
(most users never cash out / receive P2P). We quantify skew & zero-share, **winsorize** heavy tails
at p99.9, keep **log1p** copies for the linear/Cox model, and retain **indicator flags** so models
separate "structurally zero" from "low but active". Trees & RSF are invariant to monotone transforms,
so raw columns feed the GBMs.""")

code(r"""# ============================== Stage 3: skewness audit + transforms ==============================
from scipy import stats as sps
audit = []
for c in FEATURES:
    col = train[c]
    audit.append({'feature': c, 'skew': float(sps.skew(col, nan_policy='omit')),
                  'zero_share': float((col == 0).mean()),
                  'p99_over_med': float(col.quantile(0.99) / (abs(col.median()) + 1e-9))})
audit = pd.DataFrame(audit).sort_values('skew', ascending=False)
print('Most skewed:'); print(audit.head(8).to_string(index=False))

SKEWED = audit.loc[(audit['skew'] > 2) & (~audit['feature'].str.startswith('is_')), 'feature'].tolist()
caps = train[SKEWED].quantile(0.999)
for c in SKEWED:
    train[c] = train[c].clip(upper=caps[c]); test[c] = test[c].clip(upper=caps[c])
    train[c + '_log'] = np.log1p(train[c].clip(lower=0)); test[c + '_log'] = np.log1p(test[c].clip(lower=0))
LOG_MAP = {c: c + '_log' for c in SKEWED}

prefer = [c for c in ['amt_sum_90d', 'bal_mean_90d', 'in_amt_90d', 'cnt_90d'] if c in SKEWED]
extra  = [c for c in audit['feature'] if c in SKEWED and c not in prefer]
show   = (prefer + extra)[:4]; n = len(show); skew_by = audit.set_index('feature')['skew']
fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), squeeze=False)
for i, c in enumerate(show):
    sa = float(sps.skew(train[c + '_log'], nan_policy='omit'))
    axes[0, i].hist(train[c], bins=60, color='#1f6fb2'); axes[0, i].set_title(f'{c} (raw, skew={skew_by[c]:.1f})', fontsize=9)
    axes[1, i].hist(train[c + '_log'], bins=60, color='#2ca25f'); axes[1, i].set_title(f'log1p (skew {skew_by[c]:.1f}->{sa:.2f})', fontsize=9)
plt.suptitle('Pareto features: raw vs log1p'); plt.tight_layout()
plt.savefig('figures/stage3_distributions.png', dpi=130); plt.show(); plt.close()
print(f'{len(SKEWED)} features winsorized @p99.9 + log1p copies for the linear/Cox model')
""")

# ============================================================ stage 4
md(r"""## Stage 4 — The Survival Target, the 3-Horizon Reduction & Leakage Audit

**Target.** `DURATION_DAYS` ∈ [0, 91] and `EVENT_FLAG` (1 = churn observed, 0 = right-censored at
T = 91). Event rate ≈ 5 %; events cluster at 70–90 days (most churners go quiet late in the window).

**A key simplification.** All three scored horizons (30/60/90) are **≤ the censoring time 91**.
Because a censored customer (`EVENT=0`) is known active through day 91, their status at 30/60/90 is
fully known. So `SURV_PROB` at each horizon is a **fully-observed binary label** `S_h = 1[T > h]`
— no censoring correction needed for that part. This lets us train calibrated GBM classifiers
directly (great for the Brier/IBS metric) while still using a proper survival model for the ranking
(`RISK_SCORE` / c-index).

**Leakage audit.** Round 1 had a TrxID structural leak (sequential per-customer integer blocks whose
gaps encoded hidden future activity). We re-audit the new data below.""")

code(r"""# ============================== Stage 4a: target construction ==============================
y_dur = train['DURATION_DAYS'].values.astype(float)
y_evt = train['EVENT_FLAG'].values.astype(int)
# fully-observed survival labels at each horizon (all horizons <= censor time 91)
for h in HORIZONS:
    train[f'surv_{h}'] = (train['DURATION_DAYS'] > h).astype('int8')
print('event rate:', round(y_evt.mean(), 4))
print('base survival rates  P(T>h):')
base = {h: float((y_dur > h).mean()) for h in HORIZONS}
for h in HORIZONS: print(f'  S({h}) = {base[h]:.4f}')

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(y_dur[y_evt == 1], bins=46, color='#c0392b'); ax[0].set_title('DURATION_DAYS | churned (EVENT=1)')
ax[0].set_xlabel('days from Mar 31'); ax[0].set_ylabel('count')
if HAS_LIFE:   # overall Kaplan-Meier (optional — lifelines)
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(y_dur, y_evt)
    km.plot_survival_function(ax=ax[1]); ax[1].set_title('Population Kaplan-Meier S(t)')
else:          # NumPy KM fallback (steps down at event times; censored just leave the risk set)
    _o = np.argsort(y_dur); _t = y_dur[_o]; _e = y_evt[_o]; _n = len(_t)
    _S = np.cumprod(1 - _e / (np.arange(_n, 0, -1)))
    ax[1].step(_t, _S, where='post'); ax[1].set_title('Population Kaplan-Meier S(t) [NumPy]')
ax[1].set_xlabel('days from Mar 31'); ax[1].set_ylabel('S(t)')
plt.tight_layout(); plt.savefig('figures/stage4_target.png', dpi=130); plt.show(); plt.close()
""")

code(r"""# ============================== Stage 4b: LEAKAGE AUDIT — is the TrxID leak back? ==============================
# Round 1: TrxIDs were sequential per-customer integer blocks; gaps to the next block encoded the
# count of hidden post-March transactions. We test whether the new TrxIDs carry any such structure.
samp = con.execute(f"SELECT SRC_ACCOUNT, d, hr, TrxID FROM trx USING SAMPLE 40000 ROWS").df()
def hexval(s):
    try: return int(str(s).split('-')[-1], 16)
    except Exception: return np.nan
samp['idv'] = samp['TrxID'].map(hexval)
samp = samp.dropna(subset=['idv']).sort_values(['d', 'hr'])
global_corr = np.corrcoef(np.arange(len(samp)), samp['idv'].values)[0, 1]
# within-customer monotonicity of id vs time
mono_frac = []
for _, g in samp.groupby('SRC_ACCOUNT'):
    if len(g) >= 3:
        gg = g.sort_values(['d', 'hr'])['idv'].values
        mono_frac.append(np.mean(np.diff(gg) > 0))
mono = float(np.mean(mono_frac)) if mono_frac else float('nan')
print('TrxID sample format:', samp['TrxID'].iloc[0])
print(f'corr(time-order, TrxID value) = {global_corr:+.4f}   (|corr|~0 => no global counter)')
print(f'mean within-customer monotonic fraction = {mono:.3f}   (~0.5 => random, ~1.0 => time-ordered)')
LEAK_PRESENT = abs(global_corr) > 0.3 and mono > 0.9
print('VERDICT:', 'LEAK STILL EXPLOITABLE' if LEAK_PRESENT else
      'NO LEAK — TrxIDs are randomised hashes; the Round-1 exploit is patched. We model legitimately.')
del samp; gc.collect()
""")

# ============================================================ stage 5
md(r"""## Stage 5 — Survival Models

Four complementary models, all under **5-fold out-of-fold (OOF)** evaluation:

* **3-horizon GBM** (primary for `SURV_PROB`) — three LightGBM classifiers for `S(30/60/90)` with
  fully-observed labels, then **isotonic calibration** (for IBS) and **monotonicity enforcement**
  `S30 ≥ S60 ≥ S90`.
* **XGBoost `survival:cox`** — a scalable partial-likelihood model giving a smooth, finely-resolved
  `RISK_SCORE` that ranks the whole timeline (for the c-index).
* **Cox PH (lifelines)** & **Random Survival Forest (sksurv)** — fit on a subsample for hazard-ratio
  interpretation and survival-curve visuals (Stage 7 / Phase 2).

We pick the `RISK_SCORE` (among Cox-risk vs. the GBM survival blend) by **OOF c-index**, and report
**c-index** and **Integrated Brier Score** — the two competition metrics.""")

code(r"""# ============================== Stage 5a: FAST subsample ==============================
from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
if FAST:
    train = train.sample(80000, random_state=SEED).reset_index(drop=True)
    y_dur = train['DURATION_DAYS'].values.astype(float); y_evt = train['EVENT_FLAG'].values.astype(int)
N_FOLDS = 3 if FAST else 5
print('train rows for CV:', len(train), '| folds:', N_FOLDS)
""")

code(r"""# ============================== Stage 5b: multi-horizon grid + LGBM/XGB/CatBoost ensemble + Cox — OOF ==============================
# Tier 2: predict S(t) on a finer GRID (not just 30/60/90) so the RISK_SCORE can rank churn TIMING.
# Tier 1: ensemble LightGBM + XGBoost + CatBoost at each *submitted* horizon. CatBoost (ordered boosting,
#         symmetric trees) is genuinely decorrelated from LGBM/XGB, so it adds real AUC -> both metrics.
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata
# GRID, SUBMIT_IDX, exp_surv, risk_set, enforce_monotone, c_index, ibs_score live in the Helpers cell.
ENSEMBLE_CAT = False and HAS_CAT                       # CatBoost off by default (slower, ~no gain); set True to add
if ENSEMBLE_CAT:
    from catboost import CatBoostClassifier
LGB_P = dict(n_estimators=1200, learning_rate=0.03, num_leaves=63, colsample_bytree=0.8,
             subsample=0.8, subsample_freq=1, min_child_samples=60, n_jobs=N_THREADS,
             random_state=SEED, verbose=-1)
XGB_CLF = dict(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=10, tree_method='hist',
               n_jobs=N_THREADS, random_state=SEED, eval_metric='auc', verbosity=0)
CAT_P = dict(iterations=(150 if FAST else 500), learning_rate=0.05, depth=6, l2_leaf_reg=5,
             eval_metric='AUC', random_seed=SEED, thread_count=N_THREADS, verbose=0)
XGB_COX = dict(objective='survival:cox', eval_metric='cox-nloglik', tree_method='hist',
               learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
               min_child_weight=10, n_jobs=N_THREADS, seed=SEED)
XGB_AFT = dict(objective='survival:aft', eval_metric='aft-nloglik', aft_loss_distribution='normal',
               aft_loss_distribution_scale=1.0, tree_method='hist', learning_rate=0.05, max_depth=5,
               subsample=0.8, colsample_bytree=0.8, min_child_weight=10, n_jobs=N_THREADS, seed=SEED)

nG = len(GRID); dur = train['DURATION_DAYS'].values
oof_Sg = np.zeros((len(train), nG)); test_Sg = np.zeros((len(test), nG))   # LGBM grid survival probs
oof_Sx = np.zeros((len(train), 3));  test_Sx = np.zeros((len(test), 3))    # XGB probs @ submit horizons
oof_Sc = np.zeros((len(train), 3));  test_Sc = np.zeros((len(test), 3))    # CatBoost probs @ submit horizons
oof_cox = np.zeros(len(train));      test_cox = np.zeros(len(test))        # Cox risk
oof_aft = np.zeros(len(train));      test_aft = np.zeros(len(test))        # AFT predicted time-to-churn (for event-vs-event ordering)
Xte = test[FEATURES]
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
t0 = time.time()
print(f'Starting {N_FOLDS}-fold CV: {len(GRID)} LGBM + 3 XGB{"+Cat" if ENSEMBLE_CAT else ""} + Cox + AFT per fold '
      f'(progress prints after each model)...', flush=True)
for fold, (itr, iva) in enumerate(skf.split(train, y_evt)):
    Xtr, Xva = train.iloc[itr][FEATURES], train.iloc[iva][FEATURES]
    # multi-horizon LightGBM grid (survival prob at each GRID cutoff) — prints per horizon so it's never silent
    for gi, h in enumerate(GRID):
        yb = (dur > h).astype(int)
        if yb[itr].sum() in (0, len(itr)):            # degenerate (everyone survives this early cutoff)
            oof_Sg[iva, gi] = yb[itr].mean(); test_Sg[:, gi] += yb[itr].mean() / N_FOLDS; continue
        m = lgb.LGBMClassifier(**LGB_P)
        m.fit(Xtr, yb[itr], eval_set=[(Xva, yb[iva])], eval_metric='auc',
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_Sg[iva, gi] = m.predict_proba(Xva)[:, 1]; test_Sg[:, gi] += m.predict_proba(Xte)[:, 1] / N_FOLDS
        print(f'    fold {fold+1}/{N_FOLDS}  LGBM S{h}  ({time.time()-t0:.0f}s)', flush=True)
    # XGBoost + CatBoost classifiers at the 3 submitted horizons (ensemble partners)
    for j, h in enumerate(HORIZONS):
        yb = (dur > h).astype(int)
        mx = xgb.XGBClassifier(**XGB_CLF).fit(Xtr, yb[itr])
        oof_Sx[iva, j] = mx.predict_proba(Xva)[:, 1]; test_Sx[:, j] += mx.predict_proba(Xte)[:, 1] / N_FOLDS
        if ENSEMBLE_CAT:
            mc = CatBoostClassifier(**CAT_P).fit(Xtr, yb[itr], eval_set=(Xva, yb[iva]),
                                                 early_stopping_rounds=60)
            oof_Sc[iva, j] = mc.predict_proba(Xva)[:, 1]; test_Sc[:, j] += mc.predict_proba(Xte)[:, 1] / N_FOLDS
        print(f'    fold {fold+1}/{N_FOLDS}  XGB{"+Cat" if ENSEMBLE_CAT else ""} S{h}  ({time.time()-t0:.0f}s)', flush=True)
    # XGBoost Cox risk (label = time; negative => right-censored)
    ycox = np.where(y_evt[itr] == 1, np.maximum(y_dur[itr], 1), -np.maximum(y_dur[itr], 1))
    bst = xgb.train(XGB_COX, xgb.DMatrix(Xtr, label=ycox), num_boost_round=400)
    oof_cox[iva] = bst.predict(xgb.DMatrix(Xva)); test_cox += bst.predict(xgb.DMatrix(Xte)) / N_FOLDS
    # XGBoost AFT: predicts time-to-churn (right-censored => upper bound = +inf). Ranks event-vs-event timing.
    d_aft = xgb.DMatrix(Xtr)
    d_aft.set_float_info('label_lower_bound', np.maximum(y_dur[itr], 0.5))
    d_aft.set_float_info('label_upper_bound', np.where(y_evt[itr] == 1, np.maximum(y_dur[itr], 0.5), np.inf))
    baft = xgb.train(XGB_AFT, d_aft, num_boost_round=300)
    oof_aft[iva] = -baft.predict(xgb.DMatrix(Xva)); test_aft += -baft.predict(xgb.DMatrix(Xte)) / N_FOLDS  # -time => higher means churns sooner
    print(f'  fold {fold+1}/{N_FOLDS} done ({time.time()-t0:.0f}s)', flush=True)

oof_Sg = np.clip(oof_Sg, 1e-6, 1-1e-6); test_Sg = np.clip(test_Sg, 1e-6, 1-1e-6)
# submitted survival probs = mean of LGBM(grid subset) + XGB (+ CatBoost), then enforce monotone
parts_oof  = [oof_Sg[:, SUBMIT_IDX], oof_Sx] + ([oof_Sc] if ENSEMBLE_CAT else [])
parts_test = [test_Sg[:, SUBMIT_IDX], test_Sx] + ([test_Sc] if ENSEMBLE_CAT else [])
oof_S  = np.minimum.accumulate(np.mean(parts_oof, axis=0), axis=1)
test_S = np.minimum.accumulate(np.mean(parts_test, axis=0), axis=1)
# report per-model + ensemble AUC at the 90d horizon (the bottleneck) to see if CatBoost helped
from sklearn.metrics import roc_auc_score as _auc
_y90 = (dur > 90).astype(int)
print(f'S90 AUC  LGBM={_auc(_y90, oof_Sg[:, SUBMIT_IDX[2]]):.4f}  XGB={_auc(_y90, oof_Sx[:, 2]):.4f}'
      + (f'  Cat={_auc(_y90, oof_Sc[:, 2]):.4f}' if ENSEMBLE_CAT else '')
      + f'  ENSEMBLE={_auc(_y90, oof_S[:, 2]):.4f}')
# expected-survival-time risk (exp_surv defined in the Helpers cell; NumPy 1.x & 2.x safe)
oof_exp = exp_surv(oof_Sg); test_exp = exp_surv(test_Sg)
print('OOF multi-horizon grid + LGBM/XGB ensemble + Cox built.')
""")

code(r"""# ============================== Stage 5c: pick RISK_SCORE via blend sweep, calibrate, score ==============================
# c-index ~ S90 AUC for event-vs-censored (97% of pairs). The remaining gap is event-vs-event TIMING,
# which 1-S90 ranks randomly. We keep 1-S90 dominant and add a small, swept dose of a TIMING signal
# (AFT predicted time, Cox, expected-survival) to order churners by WHEN they churn. Floor = pure 1-S90.
from scipy.stats import rankdata
r90_oof  = rankdata(1 - oof_S[:, 2]);  r90_test = rankdata(1 - test_S[:, 2])
timing = {'aft': (oof_aft, test_aft), 'cox': (oof_cox, test_cox),
          'exp': (-oof_exp, -test_exp), 'sumS': ((1 - oof_Sg).sum(1), (1 - test_Sg).sum(1))}
best_ci = c_index(y_dur, y_evt, 1 - oof_S[:, 2])                  # pure 1-S90 floor
best_name, best_oof, best_test = '1 - S90', 1 - oof_S[:, 2], 1 - test_S[:, 2]
print(f'  floor  1 - S90                 c-index {best_ci:.4f}')
for tname, (to, tt) in timing.items():
    rt_oof, rt_test = rankdata(to), rankdata(tt)
    for w in [0.97, 0.93, 0.88, 0.82, 0.75, 0.65, 0.5]:          # weight on 1-S90 (rest on timing)
        blend = w * r90_oof + (1 - w) * rt_oof
        ci = c_index(y_dur, y_evt, blend)
        if ci > best_ci:
            best_ci, best_name = ci, f'{w:.2f}*S90 + {1-w:.2f}*{tname}'
            best_oof, best_test = blend, w * r90_test + (1 - w) * rt_test
    print(f'  best with {tname:4s} so far        c-index {best_ci:.4f}  ({best_name})')
print(f'chosen RISK_SCORE: {best_name}  (OOF c-index {best_ci:.4f})')
test_RISK = best_test
cand = {best_name: best_oof}; best_risk = best_name           # for the final scorecard cell

# isotonic calibration of survival probs (fit on OOF, apply to test) — improves IBS
test_Scal = test_S.copy()
for i, h in enumerate(HORIZONS):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(oof_S[:, i], (y_dur > h).astype(float))
    test_Scal[:, i] = iso.transform(test_S[:, i])
test_Scal = np.minimum.accumulate(test_Scal, axis=1)   # re-enforce monotone after calibration

ci = c_index(y_dur, y_evt, cand[best_risk], exact=True)   # one exact call for the reported number
ibs = ibs_score(y_dur, y_evt, y_dur, y_evt, oof_S)
brier = brier_at_horizons(y_dur, oof_S)
print(f'\nOOF c-index = {ci:.4f} | OOF IBS = {ibs:.4f} | Brier@30/60/90 = '
      f'{brier[30]:.4f}/{brier[60]:.4f}/{brier[90]:.4f}')
print(f'Estimated Phase-1 score ~ {comp_score(ci, ibs):.1f} / 70')
""")

# ============================================================ stage 6
md(r"""## Stage 6 — Hyperparameter Tuning (Optuna / Bayesian TPE)

We tune the **90-day survival classifier** (the hardest horizon — where almost all events live, and
the biggest lever on both IBS and the c-index). TPE searches where the surrogate expects improvement;
the best config is re-validated by OOF and compared against the default.""")

code(r"""# ============================== Stage 6: Optuna on the S(90) classifier (optional) ==============================
from sklearn.metrics import roc_auc_score
y90 = (train['DURATION_DAYS'].values > 90).astype(int)
Xtr, Xva, ytr, yva = train_test_split(train[FEATURES], y90, test_size=0.25, stratify=y90, random_state=SEED)

DO_TUNING = False    # OFF by default: tuned params only feed the SHAP plot, NOT predictions (saves ~10-15 min)
if HAS_OPTUNA and DO_TUNING:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    N_TRIALS = 8 if FAST else 15
    def objective(trial):
        p = dict(n_estimators=2000,
                 learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int('num_leaves', 31, 255, log=True),
                 min_child_samples=trial.suggest_int('min_child_samples', 20, 300, log=True),
                 colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                 subsample=trial.suggest_float('subsample', 0.5, 1.0), subsample_freq=1,
                 reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                 reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))
        m = lgb.LGBMClassifier(**p, n_jobs=N_THREADS, random_state=SEED, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(60, verbose=False)])
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    def _cb(study, trial):    # per-trial progress so the stage is never silent
        print(f'    optuna trial {trial.number+1}/{N_TRIALS}  AUC={trial.value:.5f}  best={study.best_value:.5f}', flush=True)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb], show_progress_bar=False)
    print('best params:', study.best_params)
    vals = [t.value for t in study.trials]; best = np.maximum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(1, len(vals)+1), vals, s=22, color='#1f6fb2', label='trial AUC')
    ax.plot(range(1, len(vals)+1), best, color='#c0392b', label='best so far')
    ax.set_xlabel('trial'); ax.set_ylabel('holdout AUC'); ax.set_title('Optuna (TPE) — S(90) classifier'); ax.legend()
    plt.tight_layout(); plt.savefig('figures/stage6_optuna.png', dpi=130); plt.show(); plt.close()
    LGB_BEST = {**LGB_P, **study.best_params}
else:
    print('Stage 6: tuning skipped (DO_TUNING=False) — using default LGBM params, which already drive the CV.')
    LGB_BEST = dict(LGB_P)
""")

# ============================================================ stage 7
md(r"""## Stage 7 — Explainability: SHAP, Hazard Ratios, Survival Curves

* **SHAP** on the S(90) classifier — which behaviours drive *whether the customer survives the
  quarter*, in plain language.
* **Cox hazard ratios (lifelines)** — interpretable multipliers on the instantaneous churn rate
  (the headline numbers for Phase 2).
* **Random Survival Forest curves** — show how predicted survival diverges across risk segments.""")

code(r"""# ============================== Stage 7a: SHAP on S(90) (optional) ==============================
print('Stage 7a: fitting S90 model + SHAP...', flush=True)
m90 = lgb.LGBMClassifier(**LGB_BEST).fit(train[FEATURES], (train['DURATION_DAYS'].values > 90).astype(int))
if HAS_SHAP:
    import shap
    Xs = train[FEATURES].sample(min(4000, len(train)), random_state=SEED)
    sv = shap.TreeExplainer(m90).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': np.abs(sv).mean(0)}).sort_values('mean_abs_shap', ascending=False)
    print('Top churn-timing drivers (|SHAP| on surviving past 90d):')
    print(imp.head(12).to_string(index=False))
    imp.to_csv('explainability/shap_mean_abs.csv', index=False)
    plt.figure(); shap.summary_plot(sv, Xs, show=False, max_display=15)
    plt.tight_layout(); plt.savefig('explainability/shap_summary.png', dpi=130, bbox_inches='tight'); plt.close()
else:   # fallback: LightGBM gain importance (no SHAP dependency)
    imp = pd.DataFrame({'feature': FEATURES, 'gain': m90.booster_.feature_importance('gain')}).sort_values('gain', ascending=False)
    print('shap unavailable — top drivers by LightGBM gain:'); print(imp.head(12).to_string(index=False))
    imp.to_csv('explainability/shap_mean_abs.csv', index=False)
""")

code(r"""# ============================== Stage 7b: Cox hazard ratios (optional — lifelines) ==============================
print('Stage 7b: Cox hazard ratios...', flush=True)
if HAS_LIFE:
    from lifelines import CoxPHFitter
    cox_feats = ['recency_days', 'gap_z', 'cnt_30d', 'trend_30_90', 'active_rate_30d',
                 'bal_last_over_max', 'bal_stagnation_days', 'in_cnt_90d', 'tenure_days', 'amt_mean_90d']
    cox_feats = [c for c in cox_feats if c in train.columns]
    sub = train.sample(min(60000, len(train)), random_state=SEED).copy()
    cdf = sub[cox_feats].copy()
    for c in cox_feats:                      # standardise so HRs are per-1-SD and comparable
        s = cdf[c].std() + 1e-9; cdf[c] = (cdf[c] - cdf[c].mean()) / s
    cdf['T'] = sub['DURATION_DAYS'].clip(lower=0.5).values; cdf['E'] = sub['EVENT_FLAG'].values
    cph = CoxPHFitter(penalizer=0.01).fit(cdf, 'T', 'E')
    hr = cph.summary[['exp(coef)', 'p']].rename(columns={'exp(coef)': 'HR_per_1SD'}).sort_values('HR_per_1SD', ascending=False)
    print('Cox hazard ratios (per +1 SD; HR>1 = churns SOONER, <1 = protective):')
    print(hr.round(3).to_string()); print('concordance:', round(cph.concordance_index_, 4))
    ax = cph.plot(); ax.set_title('Cox hazard ratios (log scale)')
    plt.tight_layout(); plt.savefig('explainability/cox_hazard_ratios.png', dpi=130, bbox_inches='tight'); plt.close()
else:
    print('lifelines unavailable — skipping Cox hazard-ratio plot (SHAP/gain importance covers drivers).')
""")

code(r"""# ============================== Stage 7c: Random Survival Forest curves (optional — scikit-survival) ==============================
print('Stage 7c: survival curves by risk segment...', flush=True)
if HAS_SKSURV:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    sub = train.sample(min(15000, len(train)), random_state=SEED)
    y_sub = Surv.from_arrays(sub['EVENT_FLAG'].astype(bool), sub['DURATION_DAYS'].clip(lower=0.5))
    rsf = RandomSurvivalForest(n_estimators=120, min_samples_leaf=40, max_features='sqrt',
                               n_jobs=N_THREADS, random_state=SEED).fit(sub[FEATURES], y_sub)
    risk_sub = rsf.predict(sub[FEATURES]); q = np.quantile(risk_sub, [0.1, 0.5, 0.9])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab, idx in [('low risk (p10)', np.argmin(np.abs(risk_sub - q[0]))),
                     ('median risk', np.argmin(np.abs(risk_sub - q[1]))),
                     ('high risk (p90)', np.argmin(np.abs(risk_sub - q[2])))]:
        fn = rsf.predict_survival_function(sub[FEATURES].iloc[[idx]])[0]
        ax.step(fn.x, fn(fn.x), where='post', label=lab)
    for h in HORIZONS: ax.axvline(h, ls=':', c='grey', lw=0.8)
    ax.set_title('RSF predicted survival curves by risk segment'); ax.set_xlabel('days from Mar 31'); ax.set_ylabel('S(t)'); ax.legend()
    plt.tight_layout(); plt.savefig('explainability/rsf_survival_curves.png', dpi=130); plt.show(); plt.close()
    print('RSF subsample C-index:', round(rsf.score(sub[FEATURES], y_sub), 4))
else:   # NumPy fallback: KM curves for model-defined low/median/high risk segments (uses oof risk)
    seg_risk = 1 - oof_S[:, 2]; cut = np.quantile(seg_risk, [0.33, 0.66])
    grp = np.digitize(seg_risk, cut)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for gi, lab in [(0, 'low risk'), (1, 'median risk'), (2, 'high risk')]:
        mask = grp == gi; o = np.argsort(y_dur[mask]); t = y_dur[mask][o]; e = y_evt[mask][o]; n = len(t)
        S = np.cumprod(1 - e / np.arange(n, 0, -1)); ax.step(t, S, where='post', label=lab)
    for h in HORIZONS: ax.axvline(h, ls=':', c='grey', lw=0.8)
    ax.set_title('KM survival by model risk segment [NumPy]'); ax.set_xlabel('days from Mar 31'); ax.set_ylabel('S(t)'); ax.legend()
    plt.tight_layout(); plt.savefig('explainability/rsf_survival_curves.png', dpi=130); plt.close()
    print('scikit-survival unavailable — used NumPy KM-by-risk-segment fallback.')
""")

# ============================================================ stage 8
md(r"""## Stage 8 — Business Recommendations: Time-Sensitive Retention

A survival model enables **dynamic, time-sensitive** retention — not a single risk flag but *when*
to intervene. We bucket customers by predicted 30-day churn probability and map each tier to a
timed action; the hazard tells Operations *how urgent* each outreach is.""")

code(r"""# ============================== Stage 8: retention tiers + hazard-based prioritisation ==============================
churn30 = 1 - oof_S[:, 0]; churn90 = 1 - oof_S[:, 2]
tier = pd.cut(churn30, [-1, 0.02, 0.05, 0.15, 1.0], labels=['watch', 'soon', 'urgent', 'critical'])
seg = pd.DataFrame({'tier': tier, 'churn90': churn90, 'event': y_evt, 'dur': y_dur})
summ = seg.groupby('tier', observed=True).agg(customers=('event', 'size'),
        actual_event_rate=('event', 'mean'), mean_pred_churn90=('churn90', 'mean'),
        median_duration=('dur', 'median'))
print(summ.round(3).to_string())

INTERV = {'critical': 'Same-week call + fee-waived voucher (churn imminent within ~30d)',
          'urgent':   'Push + cashback on dominant trx type within 7d',
          'soon':     'Automated re-engagement nudge / bill reminder',
          'watch':    'Standard lifecycle marketing'}
print('\nTimed interventions:')
for k, v in INTERV.items(): print(f'  {k:9s} -> {v}')

# lift of targeting the top-decile risk
order = np.argsort(-churn90); top10 = order[:len(order)//10]
lift = y_evt[top10].mean() / y_evt.mean()
print(f'\nTop-10% risk captures {y_evt[top10].sum()/y_evt.sum():.1%} of all churners (lift {lift:.1f}x).')
fig, ax = plt.subplots(figsize=(7, 4))
summ['actual_event_rate'].plot.bar(ax=ax, color='#c0392b'); ax.set_ylabel('actual churn rate'); ax.set_title('Churn rate by retention tier')
plt.tight_layout(); plt.savefig('figures/stage8_tiers.png', dpi=130); plt.show(); plt.close()
""")

# ============================================================ submission + scoring
md(r"""## Submission, OOF Scoring & Report

Write `predictions.csv` (5 columns, monotone survival probs) and report the two competition metrics
on out-of-fold predictions.""")

code(r"""# ============================== Submission ==============================
sub = pd.DataFrame({'ACCOUNT_ID': test['ACCOUNT_ID'].values,
                    'RISK_SCORE': test_RISK.astype(float),
                    'SURV_PROB_30D': test_Scal[:, 0], 'SURV_PROB_60D': test_Scal[:, 1],
                    'SURV_PROB_90D': test_Scal[:, 2]})
# guarantees: probs in [0,1], monotone decreasing, risk finite & non-negative
sub[['SURV_PROB_30D','SURV_PROB_60D','SURV_PROB_90D']] = np.minimum.accumulate(
    sub[['SURV_PROB_30D','SURV_PROB_60D','SURV_PROB_90D']].clip(0,1).values, axis=1)
sub['RISK_SCORE'] = (sub['RISK_SCORE'] - sub['RISK_SCORE'].min()).clip(lower=0)
assert (sub['SURV_PROB_30D'] >= sub['SURV_PROB_60D'] - 1e-9).all()
assert (sub['SURV_PROB_60D'] >= sub['SURV_PROB_90D'] - 1e-9).all()
assert len(sub) == len(test_ids) and sub['ACCOUNT_ID'].is_unique
sub.to_csv('predictions.csv', index=False)
print('predictions.csv written:', sub.shape)
print(sub.head().to_string(index=False))
""")

code(r"""# ============================== Final OOF scorecard ==============================
final_ci = c_index(y_dur, y_evt, cand[best_risk], exact=True)
final_ibs = ibs_score(y_dur, y_evt, y_dur, y_evt, oof_S)
br = brier_at_horizons(y_dur, oof_S)
score = comp_score(final_ci, final_ibs)
scorecard = pd.DataFrame([{'metric': 'c-index (RISK_SCORE)', 'value': round(final_ci, 4), 'points': round(max(0,(final_ci-0.5)/0.5*40),1)},
                          {'metric': 'IBS (SURV_PROB)',       'value': round(final_ibs, 4), 'points': round(max(0,(0.05-final_ibs)/0.05*30),1)},
                          {'metric': 'PHASE 1 TOTAL',          'value': '',                  'points': round(score,1)}])
print(scorecard.to_string(index=False))
print(f'\nBrier @30/60/90 = {br[30]:.4f} / {br[60]:.4f} / {br[90]:.4f}')
scorecard.to_csv('figures/scorecard.csv', index=False)
""")

code(r"""# ============================== report.pdf + presentation.pdf ==============================
with PdfPages('report.pdf') as pdf:
    fig = plt.figure(figsize=(8.3, 11.7)); fig.clf()
    txt = ('FictiPay Round 2 — Survival Model Report (Team Cybernauts)\n\n'
           f'Pipeline: DuckDB out-of-core FE over {int(sizes.iloc[0,1]):,} transactions.\n'
           f'Covariates: {len(FEATURES)} | observation 2024-01-01..03-31 | T from 2024-03-31.\n'
           f'Validation: {N_FOLDS}-fold OOF | event rate {y_evt.mean():.3f}.\n\n'
           f'Chosen RISK_SCORE: {best_risk}\n'
           f'OOF c-index = {final_ci:.4f}  ->  {max(0,(final_ci-0.5)/0.5*40):.1f}/40 pts\n'
           f'OOF IBS     = {final_ibs:.4f}  ->  {max(0,(0.05-final_ibs)/0.05*30):.1f}/30 pts\n'
           f'Estimated Phase-1 total = {score:.1f}/70\n\n'
           f'Brier @30/60/90 = {br[30]:.4f} / {br[60]:.4f} / {br[90]:.4f}\n'
           f'Base survival   S30={base[30]:.3f} S60={base[60]:.3f} S90={base[90]:.3f}\n\n'
           'Leakage audit: TrxIDs are randomised hashes (Round-1 sequential-block leak is patched);\n'
           'predictions are fully model-based and legitimate.')
    fig.text(0.08, 0.95, txt, va='top', fontsize=11, family='monospace'); pdf.savefig(fig); plt.close()
    for img in ['figures/stage4_target.png', 'figures/stage6_optuna.png',
                'explainability/shap_summary.png', 'explainability/cox_hazard_ratios.png',
                'explainability/rsf_survival_curves.png', 'figures/stage8_tiers.png']:
        if os.path.exists(img):
            fig = plt.figure(figsize=(8.3, 6)); ax = fig.add_axes([0,0,1,1]); ax.axis('off')
            ax.imshow(plt.imread(img)); pdf.savefig(fig); plt.close()
print('report.pdf written')

with PdfPages('presentation.pdf') as pdf:
    slides = [('FictiPay Round 2 — Customer Survival\nTeam Cybernauts',
               'Predicting WHEN customers churn (30-day inactivity).\nSurvival model on Jan-Mar behaviour.'),
              ('Approach', f'{len(FEATURES)} behavioural covariates -> 3-horizon GBM (SURV_PROB)\n+ XGBoost Cox (RISK_SCORE). 5-fold OOF.\nLeak audit: IDs randomised -> patched, fully legitimate.'),
              ('Result', f'OOF c-index {final_ci:.3f} | IBS {final_ibs:.4f}\nEstimated {score:.0f}/70 on Phase 1.'),
              ('What drives churn timing', 'Top: recency, gap_z (personal rhythm), 30d activity,\nbalance drain. See Cox hazard ratios & SHAP.'),
              ('Business: time-sensitive retention', 'Tiered, hazard-prioritised outreach;\ntop-10% risk captures most churners early.')]
    for title, body in slides:
        fig = plt.figure(figsize=(11, 6.2)); fig.clf()
        fig.text(0.07, 0.78, title, fontsize=22, weight='bold')
        fig.text(0.07, 0.55, body, fontsize=14, va='top'); pdf.savefig(fig); plt.close()
print('presentation.pdf written')

print('\n==================== ARTIFACTS ====================')
for fpath in ['predictions.csv', 'features.md', 'report.pdf', 'presentation.pdf']:
    if os.path.exists(fpath): print(f'  {fpath:24s} {os.path.getsize(fpath)//1024:5d} KB')
for img in sorted(glob.glob('explainability/*')): print('  ', img)
""")

# ============================================================ write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebook.ipynb')
json.dump(nb, open(out, 'w'), indent=1)
print(f'wrote {out} ({len(cells)} cells: {sum(c["cell_type"]=="code" for c in cells)} code, '
      f'{sum(c["cell_type"]=="markdown" for c in cells)} md)')
