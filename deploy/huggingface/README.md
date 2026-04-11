---
title: High-Pay US Salary Quantile Predictor
emoji: 💼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Quantile XGBoost salary predictor (P10/P50/P90)
---

# High-Paying Jobs in the US — Salary Quantile Predictor

[![GitHub](https://img.shields.io/badge/GitHub-MarwaBS/High__pay__Analysis__us-181717?logo=github)](https://github.com/MarwaBS/High_pay_Analysis_us)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost_quantile-orange)](https://github.com/MarwaBS/High_pay_Analysis_us/blob/main/MODEL_CARD.md)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://github.com/MarwaBS/High_pay_Analysis_us/blob/main/api/main.py)

Live demo of an end-to-end ML service analysing high-paying US jobs
(≥ $100K/yr). A multi-quantile XGBoost model returns **P10 / P50 / P90**
income predictions for a given demographic and occupational profile.

**The container runs two processes**:
- **FastAPI** on `localhost:8000` — the prediction service, with Redis-
  backed deterministic caching (no-op on the free tier), proxy-aware
  rate limiting, distributed drift monitoring, and precomputed O(log n)
  benchmark lookups.
- **Streamlit** on `:7860` — the dashboard you see above. The Predictor
  tab calls `POST /predict` on the local API so both surfaces share one
  prediction path.

> **This is a portfolio demo, not a deployable salary predictor.** The
> model operates on a truncated `INCTOT ≥ $100K` cohort and returns an
> honest uncertainty range, not a precise dollar estimate. See the
> `MODEL_CARD.md` in the GitHub repo for the full framing, limitations,
> and fairness discussion.

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | EDA explorer — top occupations, education distribution, gender gap within the cohort |
| **Geographic** | Choropleth of average income / location quotient / record count by state |
| **Predictor** | Enter a profile → the dashboard POSTs to the FastAPI `/predict` endpoint and renders the P10/P50/P90 quantile trio with the percentile rank inside the comparable group |
| **Model Insights** | XGBoost feature importance, permutation importance, residual plot, subgroup R² and MAE by gender and region |

## How the prediction works under the hood

1. The Predictor tab collects your inputs and builds a `PredictRequest`
   JSON payload.
2. It POSTs to `http://localhost:8000/predict` (the co-located FastAPI
   service running inside this same container).
3. The API validates the payload, checks its Redis prediction cache
   (no-op on the free tier because `REDIS_URL` is empty), precomputes
   features via a shared `pipeline.engineer_features` helper, runs
   the multi-quantile XGBoost model in one shot, and builds the
   response with percentile + group benchmarks.
4. The dashboard renders `predicted_p10`, `predicted_p50`,
   `predicted_p90`, and the percentile in the comparable cohort.

This architecture means the API is the single source of truth for
predictions — the dashboard never loads the model in-process for
scoring, so rate limiting, drift monitoring, and cache semantics all
apply uniformly to any future consumer.

## Links

- **GitHub**: <https://github.com/MarwaBS/High_pay_Analysis_us>
- **Model card**: <https://github.com/MarwaBS/High_pay_Analysis_us/blob/main/MODEL_CARD.md>
- **API docs** (if you want to see the OpenAPI spec): `https://<this-space>.hf.space/docs` is NOT exposed because Streamlit is the public surface. To inspect the API contract, see [`api/schemas.py`](https://github.com/MarwaBS/High_pay_Analysis_us/blob/main/api/schemas.py) on GitHub.

## Notes on the live environment

- First load may take ~10 seconds while both uvicorn and Streamlit come up.
- There is **no Redis cache** on the free tier — every prediction hits the model. The API falls back to a graceful no-op cache.
- There is **no persistent storage** — state is lost on container restart.
