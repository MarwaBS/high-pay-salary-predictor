# Deploying to Hugging Face Spaces

**Goal**: get the Streamlit dashboard + FastAPI service running as a single
Docker Space at
`https://<your-hf-username>-high-pay-salary-predictor.hf.space`.

**Total time**: ~15 minutes (10 min of manual clicks + 5–10 min of build time
on Hugging Face's side).

**Cost**: free. The HF Spaces "CPU Basic" tier gives 16 GB RAM and 2 vCPU, no
credit card. On the free tier the Space sleeps after inactivity and wakes on
the next visit (~30 s cold start).

---

## Prerequisites

1. A Hugging Face account. Sign up free at <https://huggingface.co/join> if
   you don't have one.
2. Git installed locally.
3. A local clone of this project (referred to below as
   `<PROJECT_DIR>` — substitute your own clone path).

---

## Step 1 — Create a Hugging Face access token

You need a **write-scope** token to push to the Space repo.

1. Go to <https://huggingface.co/settings/tokens>.
2. Click **"New token"** (top right).
3. Fill in:
   - **Name**: `high-pay-deploy` (or anything memorable)
   - **Token type**: **Write**
4. Click **"Generate a token"**.
5. **Copy the token immediately** — you will not see it again. Paste it
   somewhere temporary (a scratch file, a password manager — not the repo).
   It looks like `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.

---

## Step 2 — Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Fill in:
   - **Owner**: your HF username (e.g. `marwabs`). The Space URL will be
     `https://huggingface.co/spaces/<owner>/<space-name>` and the live app
     URL will be `https://<owner>-<space-name>.hf.space`.
   - **Space name**: `high-pay-salary-predictor` (or anything you like;
     this example assumes that name below)
   - **License**: `mit`
   - **Select the Space SDK**: click **Docker** → **Blank** (NOT "From
     template" — we're providing our own Dockerfile).
   - **Space hardware**: `CPU basic · 2 vCPU · 16 GB · FREE`
   - **Visibility**: **Public** (required for the free tier and for
     reviewers to see it without an HF account)
3. Click **"Create Space"**.
4. You land on the new Space page. It's empty except for a placeholder
   `README.md` and a default `Dockerfile`. Leave both — you'll overwrite
   them in the next step.

**Note the two URLs you now have**:

- **Space repo page** (where you push code, read build logs, edit settings):
  `https://huggingface.co/spaces/<owner>/high-pay-salary-predictor`
- **Live app URL** (what you share with reviewers):
  `https://<owner>-high-pay-salary-predictor.hf.space`

---

## Step 3 — Clone the Space repo locally

Open a terminal (PowerShell or Git Bash — either works on Windows):

```bash
cd /path/to/your/workspace   # any folder where you keep clones
git clone https://<owner>:<hf_token>@huggingface.co/spaces/<owner>/high-pay-salary-predictor hf-space-high-pay
cd hf-space-high-pay
```

Replace `<owner>` with your HF username and `<hf_token>` with the write
token from Step 1. The token is embedded in the URL so `git push` will
work without prompting.

You should now have an empty folder with just a `README.md` and the
default Dockerfile.

---

## Step 4 — Assemble the deployment files into the Space

`deploy/huggingface/assemble.sh` is the authoritative definition of what the
Space contains: it copies exactly the paths the HF Dockerfile `COPY`s, and
nothing else. From `hf-space-high-pay`:

```bash
# Paths assume you're in the hf-space-high-pay directory
PROJECT="<PROJECT_DIR>"   # substitute the absolute path to your local clone

bash "$PROJECT/deploy/huggingface/assemble.sh" "$PROJECT" .
```

`tests/test_space_snapshot.py` asserts assemble.sh satisfies every COPY source
the Dockerfile needs, so a Space assembled this way has everything the build
reads.

> **Windows note**: run this from Git Bash — `assemble.sh` needs `bash`.

### What NOT to copy

assemble.sh copies only what the image needs, so these reach the Space only if
you add them by hand. Keep them **out** — they're either private, irrelevant
to the demo, or too large:

- `private/` — **never** push this directory.
- `tests/` — not used at runtime.
- `Resources/` (~7 MB raw data), `Images/` (~11 MB), `*.ipynb` — not needed.
- `.git/`, `.github/`, `.vscode/`, `.venv/`, `__pycache__/` — clutter.
- `Dockerfile` (the original one from the project root, for docker-compose)
  — the Space uses the HF-specific Dockerfile instead.
- `docker-compose.yml`, `Makefile`, `k8s/` — wrong deployment target.

---

## Step 5 — Verify the layout

Your `hf-space-high-pay` directory should look roughly like this:

```
hf-space-high-pay/
├── Dockerfile                 # HF Spaces-specific, runs both services
├── README.md                  # Space README with YAML frontmatter
├── api/
│   ├── __init__.py
│   ├── cache.py
│   ├── drift.py
│   ├── inference.py
│   ├── main.py
│   └── schemas.py
├── scripts/
│   └── ...
├── models/
│   ├── xgb_salary_model.ubj
│   ├── xgb_premium_classifier.ubj
│   ├── baseline_stats.json
│   ├── conformal_delta.json
│   ├── feature_names.json
│   ├── group_means.json
│   └── model_metrics.json
├── Data/
│   └── cleaned_high_pay_data.csv
├── deploy/
│   └── huggingface/
│       ├── requirements-space.txt
│       └── start.sh              # Bash entrypoint
├── pipeline.py
├── config_schema.py
├── config.yaml
├── streamlit_app.py
├── pyproject.toml
└── requirements-api.txt
```

Check that **no `private/` directory** is present. `ls private/` should
return "No such file or directory". Double-check with:

```bash
git status
ls -la | grep private
```

---

## Step 6 — Push to Hugging Face

```bash
git add .
git commit -m "Initial deploy: FastAPI + Streamlit dashboard"
git push origin main
```

HF will accept the push and immediately start building the Docker image.

If git complains about `models/` or `Data/` being "large files" (>10 MB
for HF's Git LFS policy), you'll need to enable LFS:

```bash
git lfs install
git lfs track "models/*.ubj" "Data/*.csv"
git add .gitattributes
git commit -m "Track large binaries via LFS"
git push
```

Our `models/xgb_salary_model.ubj` is 588 KB and `Data/cleaned_high_pay_data.csv`
is 1.3 MB, both well under the 10 MB warning threshold, so LFS is probably
not needed.

---

## Step 7 — Watch the build

Go to your Space page at
`https://huggingface.co/spaces/<owner>/high-pay-salary-predictor`.

You should see a **"Building"** status badge and a **"Logs"** button. Click
Logs to see the Docker build output live.

**What to expect in the logs**:

1. `Step 1/19 : FROM python:3.11-slim` — base image pull (~30 s)
2. System apt-get install (~10 s)
3. `pip install -r requirements-api.txt` (~2–3 min)
4. `pip install streamlit plotly matplotlib ...` (~1–2 min)
5. `COPY` of source files (~5 s)
6. `pip install --user -e .` (~10 s)
7. **"Container is running"** — success

**Total first-build time: ~5–8 minutes.** Subsequent pushes that only
change source code will rebuild in ~30 s thanks to Docker layer caching.

**If the build fails**, scroll up in the logs to find the first red line.
The most likely failure modes:

- **Out of memory during `pip install`** — rare on CPU-basic but possible.
  Fix: go to Space **Settings → Hardware** → upgrade temporarily to
  "CPU upgrade" for the first build, then switch back.
- **`models/xgb_salary_model.ubj` not found** — assemble.sh didn't run to
  completion in Step 4. Re-run it and push again.
- **`Data/cleaned_high_pay_data.csv` not found** — same fix.

---

## Step 8 — Open the live demo

Once the status badge flips to **"Running"**:

**Live URL**: `https://<owner>-high-pay-salary-predictor.hf.space`

Bookmark this. Share it on your CV, LinkedIn, portfolio page.

Expected first-load UX:

1. Browser hits the URL → Streamlit's page loads (~1–2 s)
2. Sidebar filters populate (~1 s)
3. Click the **Predictor** tab → fill in a profile → click Predict
4. The dashboard POSTs to `http://localhost:8000/predict` inside the
   container → the FastAPI service responds in 10–50 ms (no network
   round-trip) → the P10/P50/P90 trio renders

If the Predictor tab shows an error like "Could not reach the API", it
means the API hasn't finished booting yet. Refresh after ~10 seconds.

---

## Step 9 — Add the live demo badge to the main README

Once the Space is running, add this to the top of the **project
`README.md`** (the one on GitHub, not the Space one):

```markdown
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-on%20Hugging%20Face-yellow)](https://<owner>-high-pay-salary-predictor.hf.space)
```

Commit and push to GitHub as usual. Now any recruiter browsing the
GitHub repo has a one-click path to the live demo.

---

## Updating the Space after code changes

Once the Space exists, updates are automatic. `.github/workflows/deploy.yml`
runs on every push to `main`: it clones the Space, overlays the snapshot with
`deploy/huggingface/assemble.sh`, and pushes if anything changed. HF then
rebuilds (~30 s for code-only changes, ~3 min if deps changed). Model updates
ship the same way — commit the retrained `models/` artefacts to `main`.

This requires a **write-scoped** HF token stored as the `HF_TOKEN` repository
secret (GitHub → Settings → Secrets and variables → Actions). The workflow
validates the token first and fails with instructions if it is missing or
read-scoped.

A weekly `space-drift` job in the same workflow re-runs the assembly against
the live Space read-only; any diff means the Space is no longer serving
`main`, and the job fails.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails at `pip install xgboost` with "killed" | OOM on CPU basic | Upgrade to CPU upgrade for the first build |
| Container keeps restarting | `start.sh` isn't executable | `chmod +x start.sh && git commit -am "chmod" && git push` |
| Dashboard loads but Predictor tab errors | API didn't finish booting | Wait 10 s, refresh |
| Dashboard error: "Could not reach the API" | `API_BASE_URL` mis-set | Verify `ENV API_BASE_URL=http://localhost:8000` in the Dockerfile |
| 404 on the Space URL | Build not finished | Check Space page for "Building" status |
| "This Space is sleeping" notice | Long inactivity | Click "Restart this Space" — it wakes in ~10 s |

---

## What to commit back to the main project

**Only** commit the `deploy/huggingface/` directory (Dockerfile, start.sh,
README.md, requirements-space.txt, assemble.sh, this DEPLOY.md) to the main
GitHub project. That's the deployment tooling. The Space repo itself is
separate and stays local (or gets pushed only to HF).

**Never** commit your HF access token to either repo.
