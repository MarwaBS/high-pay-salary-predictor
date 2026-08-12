# Model Card — US High-Pay Salary Quantile Model

## Model Details

| Field | Value |
|---|---|
| **Name** | XGBoost multi-quantile salary predictor |
| **Version** | 2.0.0 |
| **Type** | Gradient-boosted quantile regression (XGBoost `reg:quantileerror`, alphas = [0.10, 0.50, 0.90]) |
| **Output** | P10 / P50 / P90 dollar predictions per request — not a point estimate |
| **Artefact** | `models/xgb_salary_model.ubj` (XGBoost native binary, no pickle) |
| **Training script** | `scripts/train_quantile.py` |
| **Config** | `config.yaml` |

> **Change in 2.0.0.** Previous versions framed the task as a point-estimate
> regression of individual income. Within the $100K+ cohort, individual
> variance is dominated by unobserved factors (equity, bonuses, tenure,
> specific employer), so a point estimator cannot produce a useful R² on
> this data. The 2.0.0 model instead returns a calibrated quantile interval
> (P10/P50/P90) and is scored on empirical coverage rather than squared error.

## Intended Use

Given a demographic and occupational profile within the **$100K+ US
cohort**, return a calibrated income range (P10, P50, P90) the worker
can use as a directional benchmark. Intended for exploratory analysis
and portfolio demonstration — **not** for employment decisions,
compensation benchmarking, or any consequential use.

## Training Data

| Source | Description |
|---|---|
| US Census ACS/CPS microdata | Individual income, demographics, education, occupation |
| BLS OEWS | State × occupation employment, location quotient, hourly/annual mean wage |

- **Population**: workers with reported annual income ≥ $100 K
- **Geography**: all 50 US states
- **Features used**: 10 (see below)

### ⚠️ Data-prep caveat

`notebooks/high_pay_jobs_data_cleaning.ipynb` double-filters the cohort:
- BLS rows are kept only if `A_MEAN ≥ $100K` or `H_MEAN ≥ $48` (cell 14)
- Census rows are kept only if `INCTOT ≥ $100K` (cell 21)
- The two are then inner-joined on `(OCC_CODE, STATE)` (cell 9)

This truncation removes occupation-wage signal. The shipped
point estimator reaches R² ≈ 0.026 on held-out test; how much of that is
attributable to the truncation has not been measured, and no ceiling is claimed.
The quantile model still produces useful output because the P10/P50/P90
spread itself is informative.

**Gap 1 remediation — phases.**

- **Phase 1 (shipped)**: a premium-tier binary classifier head is trained
  alongside the quantile regressor by `scripts/train_quantile.py` on the
  same engineered feature matrix. The label is
  `Annual Income ≥ config.model.premium_threshold` (default `$150,000`),
  chosen because it gives a roughly 40/60 positive rate on the existing
  cohort — a well-defined, supportable binary task on the data that is
  already in the repo. The classifier is served on every `/predict`
  response as `p_above_premium_threshold`, letting callers answer "is
  this profile likely to clear the premium bar at all?" in addition to
  the quantile interval "given that it does, what's the range?".

- **Phase 2 (deferred)**: a true unfiltered `≥ $100K` membership
  classifier on the full Census microdata remains future work. The raw
  IPUMS Census export is not in the repo (fetching it requires an
  IPUMS API key), so Phase 2 is explicitly blocked on a data fetch,
  not on modelling. When the raw file lands, Phase 2 becomes a
  ~2-hour follow-up to this trainer.

## Features

Feature set is unchanged from v1.0.0 — only the training objective changed.

| Feature | Type | Source | Notes |
|---|---|---|---|
| `Age` | int | Census | |
| `Education_Ord` | int (1–4 ordinal) | Census → `config.yaml` mapping | |
| `Gender_Bin` | binary (1=Male, 0=Female) | Census | |
| `Region_Code` | int (0–3) | Derived from state → US Census region | |
| `Employment` | float | BLS OEWS | |
| `Location Quotient` | float | BLS OEWS | |
| `Jobs per 1000` | float | BLS OEWS | |
| `Hourly Mean` | float | BLS OEWS | `Annual Mean Wage` dropped (VIF ≈ 2.3×10⁷, corr = 1.0000 to 4 dp) |
| `Occ_Mean_Income` | float | Derived from **training split only** | Target-derived salary prior, frozen after fit |
| `State_Mean_Income` | float | Derived from **training split only** | Target-derived salary prior, frozen after fit |

**The last two features are priors built from the target.** They are the mean
income of the occupation and of the state, computed on training rows only and
frozen into `models/group_means.json` for test and inference. That protocol is
what keeps them out of train/test leakage, and
`tests/test_integration.py::TestSplitThenEngineer::test_no_occ_mean_leakage` holds it.

It is not only a leakage question. It sets what the model claims: at inference
the caller supplies an occupation and a state, and the server injects the
historical mean income for that pair. So the model answers "given that people in
this occupation and state earned about X, what should this person earn?", not
"what is this person worth from their own attributes alone". A pair absent from
training falls back to the global mean of the group means, and performance on
genuinely unseen occupations or states is **not measured** anywhere in this repo.
## Training Objective

```
objective = "reg:quantileerror"
quantile_alpha = [0.10, 0.50, 0.90]
target = log1p(Annual Income)
```

A single XGBoost model outputs all three quantiles simultaneously. At
inference the raw `(n, 3)` output is back-transformed via `expm1` into
dollar space. Hyperparameters come from `config.yaml` and are chosen by
`scripts/tune.py`, which scores candidates on leakage-free 5-fold pinball loss
over the training split only. The committed study
(`models/tuning_study.json`, seed 42, 60 trials) found nothing better than the
shipped values, so they were retained. Read that as a tie, not a win: the best
candidate was 0.81 worse on a paired per-fold standard error of 35.0
(t = 0.02, p = 0.98), and the top seven candidates span 17 of loss, so this
region of the space is flat. The 4,006 total spread comes from the poor
candidates. `tests/test_hyperparameter_provenance.py` binds config, study and
search space together.

## Performance

Measured on a held-out 20% test split (2,051 rows). Retrain date
shown in `models/model_metrics.json::train_date`.

### Quantile metrics (the real SLO)

| Metric | Value | What it means |
|---|---|---|
| 80% coverage — raw quantiles | ~0.77 | Fraction of test targets inside the raw `[P10, P90]`. Under-covers the 0.80 target by ~3 pts. |
| 80% coverage — **served (cross-conformal)** | **~0.79** | The API widens the interval by a conformal margin (below), closing most of the shortfall against the 0.80 target. |
| Median PI width — served | ~$115K | ~3% wider than the raw interval; the cost of closing the coverage gap. |
| Conformal margin (log space) | ~0.010 | Symmetric widening added to P10/P90; estimated by 5-fold cross-conformal on train (§ below). |
| Quantile crossings | **0** | Number of test rows where P10 > P50 or P50 > P90. Must be zero. |
| P10 pinball loss | ~$6.6K | Quantile loss at α=0.10. |
| P50 pinball loss | ~$25K | Quantile loss at α=0.50 (equals `0.5 × MAE`). |
| P90 pinball loss | ~$22K | Quantile loss at α=0.90. |

### Provenance & reproducibility

Each artefact carries a `model_version` of the form
`{service_version}+{git_sha}.{data_sha256_prefix}`, read from
`models/model_metrics.json` rather than quoted here: the weekly retrain changes
the git SHA, so any value written into this page is stale by the next Monday.
The **git SHA is the exact commit the metrics file was generated at** — either
a scheduled `train.yml` run on `main`, or a working-branch commit whose
regenerated metrics land via PR. Because PRs land via **squash-merge** (which
creates a new commit with a different SHA), the recorded commit is a real,
still-fetchable object that is generally **not an ancestor of `main`**. Do not expect
`git checkout <git_sha>` from a shallow/gc'd clone to succeed. Reproducibility
does **not** depend on checking out that commit: it rests on the committed
artefacts, the exact-version `requirements-lock.txt`, the fixed training seed
(`config.yaml::model.random_state`), and the `data_sha256` prefix that pins the
input CSV — same code + same data + same seed reproduce the same artefact bytes.
They do not reproduce the same `model_version`, because it carries the git SHA of
the commit that trained: the `data_sha256` binds a metric set to its input, the
git SHA identifies the source revision. `tests/test_model_version.py` enforces
the version *shape*.

### Point-estimate metrics (backward compat, P50 column)

| Metric | Value | Honesty note |
|---|---|---|
| Test R² (P50) | ~0.026 | P50 under a quantile objective is the median-minimiser, not the mean-minimiser, so R² (which scores means) is a weak fit-statistic for this model. The real SLO is quantile coverage above. |
| Test MAE | ~$50K | |
| Test RMSE | ~$108K | |
| CV R² (5-fold, train only, dollar space) | ~0.022 ± 0.017 | Leakage-free per-fold target encoding; close to test R² — no overfitting, no space mismatch. |

### CV alignment

CV is computed **only on the training set**, in **dollar space**, using
a fresh fold model — exactly the same space as the test metric above.
This is enforced by `tests/test_pipeline.py::TestModelPrediction::test_saved_cv_matches_test`.

### Stability across seeds

The headline numbers above come from one train/test split. To show they
are not a lucky split, `scripts/train_quantile.py` re-runs both heads
across 5 seeds and records mean ± std in `model_metrics.json`:

| Metric | Mean ± std (5 seeds) |
|---|---|
| P50 R² | ~0.017 ± 0.010 |
| 80% coverage | ~0.782 ± 0.011 |
| Classifier ROC-AUC | ~0.696 ± 0.008 |
| Classifier Brier | ~0.213 ± 0.001 |

The tight std bands confirm the metrics are stable across splits, not
single-split artefacts — including the honest one: the near-zero R² is a
**consistent** feature-ceiling result, not noise.

### Serving latency

`tests/test_performance.py` drives 100 sequential in-process `POST /predict`
calls (FastAPI `TestClient`, excludes network/proxy) and fails the build if the
nearest-rank p99 exceeds the 200 ms SLO. No absolute millisecond figure is
published here: it would describe the machine that ran it, and nothing in this
repo regenerates it.

The hot path stays off the DataFrame by moving every per-request lookup to
an O(1) dict get / O(log n) binary search precomputed at startup
(`build_benchmark_lookup`, `build_bls_defaults_lookup`, and the fallback
means) — the hot path performs no DataFrame scans.

### Premium-tier classifier head (Gap 1 Phase 1)

A separate XGBoost binary classifier is trained in the same pass with
`objective="binary:logistic"` and lighter hyper-parameters than the
regressor (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`).
Target: `Annual Income ≥ config.model.premium_threshold` (default
`$150,000`).

**No `scale_pos_weight`.** At the ~40/60 class balance the imbalance is
mild, and the API serves this output to callers as a probability
(`p_above_premium_threshold`) rather than a ranking, so the head is trained
unweighted. This is a design choice, not a measured one: the weighted
variant was never run, so the trade-off is asserted from the class balance
alone. The Brier score below beats the constant-base-rate predictor, which
is skill — the repo computes no reliability curve, so it is not a
calibration measurement.

At HEAD, on the held-out test split:

| Metric | Value | What it means |
|---|---|---|
| Positive rate (test) | ~0.39 | Fraction of the test cohort earning ≥ `$150,000`. |
| ROC-AUC | ~0.67 | Discrimination across the full threshold sweep. |
| PR-AUC | ~0.55 | Precision-recall AUC — more informative than ROC on the ~40% positive rate. |
| F1 @ 0.5 | ~0.50 | Balanced F1 at the default decision threshold. |
| **Brier score** | **~0.218** vs **0.237** no-skill | Proper score on the served probability — lower is better; beats the constant-base-rate predictor. |
| Subgroup ROC-AUC | 0.64–0.70 across Gender / Region (min 0.64, max 0.70) | No tracked slice fell to chance. Above 0.5 is a collapse check, not evidence of parity. |

**Baselines — does the GBM earn its place?** Recorded in
`model_metrics.json`, same split:

| Baseline | Value | Verdict |
|---|---|---|
| Majority-class accuracy | ~0.61 | XGBoost accuracy (~0.65) beats it, but only modestly. |
| Logistic-regression ROC-AUC | ~0.68 | **On the shipped split the head trails (0.674 vs ~0.68). Refit across the same five splits the two are ~0.696 and ~0.690, a gap smaller than either spread, so neither ranks better.** |

The honest conclusion: on this feature set the gradient-booster buys
nothing over linear logistic regression — the signal ceiling is the
**features**, not the model. XGBoost is kept for serving consistency
(same `.ubj` format as the regressor, no pickle), not for an accuracy
lift, and that trade-off is stated rather than hidden. A heroic 0.9
ROC-AUC on this double-filtered cohort would be a sign of leakage, not
skill. Phase 2 (raw `≥ $100K` membership against the unfiltered Census
cohort) is where real separability gain lives.

## Subgroup Performance

Per-group empirical 80% coverage is tracked in
`models/model_metrics.json::subgroup_coverage_80`.
`tests/test_pipeline.py::TestModelPrediction::test_subgroup_coverage_within_band` holds every slice
inside **[0.60, 0.95]**. That band is a collapse guard, not a guarantee of equal
coverage: a slice could fall from 0.77 to 0.61 and still pass. At HEAD the spread
is 0.73–0.80 across `Gender` and `Region` — narrower than the v1
point-estimator R² gap.

The quantile reframe does not directly close the subgroup gap in this
dataset because the data-prep truncation affects both subgroups. The
gap is an argument for the data-prep rewrite follow-up, not a model tweak.

## Prediction Interval

The API endpoint `POST /predict` and the Streamlit dashboard now return:

| Field | Description |
|---|---|
| `predicted_p10` | 10th-percentile salary prediction (low end of 80% PI) |
| `predicted_p50` | Median prediction (point estimate for back-compat) |
| `predicted_p90` | 90th-percentile salary prediction (high end of 80% PI) |
| `predicted_salary` | Alias for `predicted_p50`, kept for v1 clients |
| `prediction_interval_low` / `prediction_interval_high` | Same as `p10` / `p90` |

**Cross-conformal calibration.** The raw quantile interval under-covers its
nominal 80% by ~3 points, so the served `p10`/`p90` are widened symmetrically
in log space by a conformal margin. The margin is estimated by 5-fold
cross-conformal (CQR) on the training set — each fold scores the held-out rows
with `max(q_lo − y, y − q_hi)` and the margin is the 0.80 quantile of the
pooled scores (with the standard `(n+1)` small-sample lift). Because the fold
models differ from the served full-data model, coverage is approximate —
validated empirically at ≈0.80 on the held-out test set. The shipped model
still trains on all of train and its bytes are unchanged. The margin is persisted to
`models/conformal_delta.json` (content-addressed alongside the other
artefacts) and applied at serve time; P50 is never shifted. Result: served
coverage 0.7942 against the 0.80 target, at ~3% wider intervals.

Quantile crossings (P10 > P90) are clamped defensively inside
`api/inference.build_response`, so clients never see an inverted range
even if XGBoost emits one at a decision boundary. The raw crossing rate
is also exported on `/metrics` as `salary_quantile_crossings_total`, so a
rising rate — a model-health signal — is observable rather than silently
corrected.

## Limitations and Biases

1. **Binary gender**: the training data contains only "Male" / "Female"
   labels from Census CPS coding. Non-binary identities are not
   represented. The model cannot make predictions for genders outside
   this binary.

2. **Truncated cohort**: as noted in the data-prep caveat, the model is
   trained on a double-filtered slice of the population. It is
   well-defined *within* the $100K+ cohort but cannot answer "will
   this person earn more than $100K" — use a different model for that.

3. **Geographic coverage**: US data only.

4. **Temporal drift**: BLS OEWS and Census data are point-in-time
   snapshots. The Redis-backed drift monitor (`/drift` endpoint)
   aggregates observations cluster-wide so drift alerts are reliable
   across a multi-replica Deployment.

5. **Unobserved confounders**: equity compensation, bonuses, years of
   experience, specific employer, and negotiation history drive large
   income differences the model cannot capture. The quantile spread
   reflects this uncertainty honestly rather than pretending it away.

6. **Fairness**: group-level income disparities in the training data
   (by gender, region, occupation) are reflected in the quantile
   intervals. The model does not correct for historical discrimination
   embedded in wages. Per-subgroup calibration is computed at train time
   (`model_metrics.json::subgroup_coverage_80`) and gated by
   `tests/test_pipeline.py::TestModelPrediction::test_subgroup_coverage_within_band`, so a
   fairness collapse in any tracked slice fails the build.

## How to Retrain

```bash
# The only trainer — multi-quantile XGBoost, no MLflow / Optuna.
python -m scripts.train_quantile
```

Writes artefacts to `models/` and metrics to `models/model_metrics.json`.
The test suite picks up changes automatically — if the quantile coverage
drifts outside `[0.72, 0.88]` or any crossings appear,
`tests/test_pipeline.py::TestModelPrediction::test_saved_metrics_within_expected_range` will
fail loudly.
