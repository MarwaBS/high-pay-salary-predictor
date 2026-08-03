# Design Decisions

Decisions taken from this file's creation forward, with the evidence that
settled each one. Earlier choices are not reconstructed here — a record written
after the fact is a justification, not a decision log.

Each entry states what was decided, what was measured, and what would reverse
it. An entry with no reversal condition is an opinion, not a decision.

---

## D-001 — The shipped XGBoost hyper-parameters are retained, not re-tuned

**Decided:** keep `n_estimators 169`, `max_depth 3`, `learning_rate 0.045`,
`subsample 0.741`, `colsample_bytree 0.829`, `reg_lambda 9.88`.

**Why it came up:** none of them had a producer. No script in the repo emitted
them and no record said where they came from, so they could not be re-derived,
defended, or re-run against new data.

**Measured:** `scripts/tune.py` searched a six-dimensional space by seeded
random sampling — 60 trials, seed 42 — scoring mean dollar-space pinball loss
under 5-fold CV with per-fold target encoding, on the training split only. No
candidate beat the incumbent's 17,524.88; the best was 17,525.70.

**Read it as a tie, not a win.** That 0.81 gap sits on a paired per-fold
standard error of 35.0 (t = 0.02, p = 0.98), and the top seven candidates span
17 of loss. The region is flat. The 4,006 spread across the whole study comes
from the poor candidates, not from any signal near the optimum.

**The scores are also build-dependent, which settles it.** The same parameters
and seed produce 17524.88 on the machine that recorded the study, 17541.41 under
Linux/CPython 3.11 and 17544.54 under 3.12 — XGBoost's float reductions differ by
build. That 19.7 spread is **24x the 0.81 margin** between the incumbent and the
best candidate. The ranking of two configurations this close is therefore not a
portable fact, and no amount of re-running settles it. Only the absolute scores
are recorded here; the conclusion drawn from them is "nothing beat the incumbent
by more than noise", not "the incumbent is optimal".

**What would reverse it:** a study that beats the incumbent by more than the
paired standard error, or a change to the input data — `models/tuning_study.json`
records the `data_sha256` it was run against, and
`tests/test_hyperparameter_provenance.py` fails if config, study, search space
and dataset stop agreeing.

**Evidence:** `models/tuning_study.json`, `scripts/tune.py`,
`tests/test_hyperparameter_provenance.py`.

---

## D-002 — `GET /drift` requires a key and spends the prediction budget

**Decided:** `/drift` authenticates with `X-API-Key` when `API_KEY` is set and
draws on the same per-IP `RATE_LIMIT` as `/predict`.

**Why it came up:** with `API_KEY` set, `POST /predict` returned 401 while
`GET /drift` returned 200 and disclosed aggregate production traffic — per-feature
means of live requests, including `Gender_Bin` and `Region_Code`.

**Measured:** 40 consecutive anonymous calls all returned 200 before the change.
After it: no key 401, wrong key 401, correct key 200, and `RATE_LIMIT=5/minute`
gives `[200×5, 429×3]`. Authentication resolves before the limiter, so an
unauthenticated caller cannot drain a keyed operator's budget.

**Breaking for** any caller that read `/drift` anonymously.

**What would reverse it:** a deployment that needs anonymous drift reads would
have to state how the traffic summary stops being sensitive.

**Evidence:** `tests/test_api_security.py::TestDriftReportIsProtected`.

---

## D-003 — `/metrics` carries the same key as `/predict` and `/drift`

**Decided:** `/metrics` authenticates with `X-API-Key` whenever `API_KEY` is set.

**Why it came up:** it published a per-route request counter equal to the
observation count `/drift` reports. Authenticating two routes while a third
serves the same figure is not a policy, it is a gap — and it was recorded here
as an open residual, which is not a state a security inconsistency gets to stay in.

**Measured:** no key 401, wrong key 401, correct key 200 serving
`salary_quantile_crossings`; an open deployment (`API_KEY` unset) is unchanged at
200. Removing the dependency turns two tests red.

**Cost, stated:** `k8s/api-deployment.yaml`'s `prometheus.io/*` annotations carry
no headers, so the scrape job must now supply the key in its `scrape_config` or
it will collect 401s. That is written into the manifest beside the annotations.
Reopening the endpoint to keep scraping simple was the alternative and is
rejected: it would re-expose through telemetry exactly what D-002 withholds.

**What would reverse it:** moving telemetry to a sidecar or mesh that never
exposes the series on the public listener would make the key unnecessary.

**Evidence:** `tests/test_api_security.py::TestMetricsEndpointIsProtected`.


## D-004 — Two heads, and the classifier scoped inside the high-pay cohort

**Decided:** `scripts/train_quantile.py` trains a multi-quantile regressor and a
premium-tier classifier in one pass, and the classifier's label is
`Annual Income >= premium_threshold` *within the existing $100K+ cohort*.

**Why two heads:** inside that cohort, individual income has extreme
within-group variance driven by unobserved factors — equity, bonuses, tenure,
employer. No point estimator resolves that, so the regressor returns a
calibrated interval. The classifier answers a different question: is the premium
tier plausible at all for this profile? A caller needs both.

**Why the narrow label:** it is a supportable binary task on the data that is
actually in the repo (roughly 40/60 balance, see `models/model_metrics.json`). A
broader "above the $100K line at all?" membership classifier would need the
*unfiltered* IPUMS microdata — a separate fetch behind an IPUMS API key, not a
file in `Data/` — so it is a follow-up, not an omission.

**No MLflow / Optuna.** The trainer stays lean enough to run on a CI worker
without an experiment-tracking stack. Hyper-parameters are pinned in
`config.yaml` and chosen by `scripts/tune.py`; `models/tuning_study.json` is the
record. See D-001.

**Measured:** the narrow label is a usable task on the shipped data, not a
degenerate one: positive rate 0.3974 train / 0.3852 test, Brier 0.2177 against a
base rate of 0.2368, accuracy 0.6543 against a majority-class 0.6148. **It does
not beat every reference:** ROC-AUC is 0.6735 against a logistic baseline of
0.68 — the head is calibrated better than the base rate but does not out-rank a
linear model on the same features. The regressor is scored separately on
quantile coverage and crossings, which a single head could not report.

**What would reverse it:** the unfiltered IPUMS microdata landing in `Data/`,
which makes the broader "above the $100K line at all?" label supportable and
turns the classifier's cohort scoping from a data limit into a choice. The head
is retired, not rescoped, if its Brier stops beating the base rate — the
ranking gap against the logistic baseline is already recorded under Known gaps.

**Evidence:** `scripts/train_quantile.py`, `tests/test_classifier.py`,
`models/model_metrics.json` (`classifier_*` keys).

---

---

## Known gaps

Carried deliberately, not overlooked. Each states why it is open and what would
close it, so a reader does not have to infer the difference between a decision
and an omission.

- **The premium-tier classifier does not out-rank a linear baseline.** ROC-AUC
  0.6735 against `classifier_baseline_logreg_roc_auc` 0.68, both in
  `models/model_metrics.json`. It is kept because it is calibrated (Brier 0.2177
  vs base rate 0.2368) and answers a question the regressor cannot, not because
  it ranks best. `tests/test_classifier.py` enforces the no-skill floor and the
  base-rate Brier, so nothing fails on this gap; closing it means either beating
  the logistic reference or replacing the head with it.
- **The tuning study's absolute scores are not portable across builds.** See
  D-001: the observed cross-build spread is 24x the margin the study turns on.
  `tests/test_hyperparameter_provenance.py` therefore re-derives the incumbent
  score against a 1% relative tolerance rather than exactly, which catches a
  fabricated score but cannot catch one hidden inside build noise. A forgery
  that flips the study's conclusion is caught instead by the retained-value
  chain, which compares `config.yaml` against whichever parameters the study
  says won — verified by mutation.

- **The weekly retrainer runs an interpreter no measurement covers.**
  `train.yml` pins CPython 3.13. The only cross-build measurement here is of the
  tuning study's CV score — 0.11% across three builds, two of them 3.11 and 3.12
  and the third a machine whose interpreter is unstated — so nothing covers 3.13,
  and nothing measures the published metrics across builds at all. The enforced 1%
  is a round bound comfortably above that 0.11%, not a figure derived from it.
  `requirements-lock.txt` pins the library set, not the interpreter. Closing it
  means pinning the trainer to a measured interpreter, a release-process decision.

- **The classifier head's six hyper-parameters have no producer.**
  `config.yaml` sets `classifier_n_estimators 200`, `classifier_max_depth 4`,
  `classifier_learning_rate 0.05`, `classifier_subsample 0.85`,
  `classifier_colsample_bytree 0.85` and `classifier_reg_lambda 1.0`.
  `scripts/tune.py` searches the six regressor keys only, so what D-001 says of
  the regressor before it was tuned is still true here: no script emits them and
  no record says where they came from. No justification is offered for the values,
  because one written now would be written after the fact. Closing it means
  extending the search to the classifier head and running it — which can force a
  retrain, so it is a scheduled action rather than a tidy-up.

- **The paired standard error, t and p in D-001 have no committed producer.**
  `models/tuning_study.json` records per-trial means only, so the per-fold losses
  those statistics come from are not in the repo. The figures reproduce — anyone
  can re-run the CV for both parameter sets — but nothing regenerates them on
  demand. Closing it means having `scripts/tune.py` record per-fold losses.

- **`scripts/tune.py` is the least-covered module (~53%).** Its `main()` is
  exercised end-to-end by `tests/test_hyperparameter_provenance.py`, but through
  a subprocess, which coverage cannot instrument. The behaviour is tested; the
  lines are not counted. Closing it means calling `main()` in-process.

- **`requirements-lock.txt` still pins the notebook-only packages.** The split in
  `requirements.txt` does not extend to the lock, because the lock is the
  reproducibility contract the trainer installs. Regenerating it is a deliberate
  act that has to be followed by a retrain and a comparison of the published
  metrics, not a side effect of a dependency tidy-up.

- **Private repository names: closed, and guarded.** A sweep of `.gitignore`,
  `.trivyignore` and every published commit message on `main` returns no hit, by
  the digest matcher or the external ban-list. `tests/test_private_names_absent.py`
  blocks re-entry. Recorded because the guard, not the absence, is the deliverable.
