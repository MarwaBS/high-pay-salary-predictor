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

**Residual, unresolved:** `/metrics` is unauthenticated and publishes a
per-route request counter equal to `/drift`'s observation count.
`k8s/api-deployment.yaml` scrapes it, so keying it breaks scraping; a
NetworkPolicy is the usual answer. Not decided here.

**Evidence:** `tests/test_api_security.py::TestDriftReportIsProtected`.
