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
