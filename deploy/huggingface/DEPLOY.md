# Deploying to Hugging Face Spaces

**Goal**: get the Streamlit dashboard + FastAPI service running as a single
always-on Docker Space at
`https://<your-hf-username>-high-pay-salary-predictor.hf.space`.

**Total time**: ~15 minutes (10 min of manual clicks + 5–10 min of build time
on Hugging Face's side).

**Cost**: free. The HF Spaces "CPU Basic" tier gives 16 GB RAM and 2 vCPU,
always-on, no credit card.

---

## Prerequisites

1. A Hugging Face account. Sign up free at <https://huggingface.co/join> if
   you don't have one.
2. Git installed locally.
3. The deployed project at
   `C:\Users\marwa\OneDrive\Bureau\Deployed_project\High_pay_Analysis_us`
   (that's where we've been working).

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
cd "C:/Users/marwa/OneDrive/Bureau"
git clone https://<owner>:<hf_token>@huggingface.co/spaces/<owner>/high-pay-salary-predictor hf-space-high-pay
cd hf-space-high-pay
```

Replace `<owner>` with your HF username and `<hf_token>` with the write
token from Step 1. The token is embedded in the URL so `git push` will
work without prompting.

You should now have an empty folder with just a `README.md` and the
default Dockerfile.

---

## Step 4 — Copy the deployment files into the Space

From `hf-space-high-pay`, copy the following **from the deployed project**:

```bash
# Paths assume you're in the hf-space-high-pay directory
PROJECT="C:/Users/marwa/OneDrive/Bureau/Deployed_project/High_pay_Analysis_us"

# 1. HF-specific files — these replace the Space defaults
cp "$PROJECT/deploy/huggingface/Dockerfile"      ./Dockerfile
cp "$PROJECT/deploy/huggingface/README.md"       ./README.md
cp "$PROJECT/deploy/huggingface/start.sh"        ./start.sh
cp "$PROJECT/deploy/huggingface/.dockerignore"   ./.dockerignore

# 2. Python source + config
cp -r "$PROJECT/api"               ./api
cp -r "$PROJECT/scripts"           ./scripts
cp    "$PROJECT/pipeline.py"       ./
cp    "$PROJECT/config_schema.py"  ./
cp    "$PROJECT/config.yaml"       ./
cp    "$PROJECT/streamlit_app.py"  ./
cp    "$PROJECT/pyproject.toml"    ./
cp    "$PROJECT/requirements-api.txt" ./
cp    "$PROJECT/requirements.txt"     ./
cp    "$PROJECT/LICENSE"              ./

# 3. Runtime artefacts (model + data) — baked into the image, no mounts
cp -r "$PROJECT/models"            ./models
cp -r "$PROJECT/Data"              ./Data

# 4. Optional: geospatial shapefile for the choropleth tab
cp -r "$PROJECT/us_state"          ./us_state

# 5. The deploy/ folder itself — so the Dockerfile's COPY of
#    deploy/huggingface/start.sh resolves at build time.
mkdir -p ./deploy/huggingface
cp "$PROJECT/deploy/huggingface/start.sh" ./deploy/huggingface/start.sh
```

> **Windows note**: if you're on plain PowerShell, use `Copy-Item -Recurse`
> and `Copy-Item` instead of `cp -r` / `cp`. Git Bash supports the Unix-style
> commands above.

### What NOT to copy

Keep these **out** of the Space — they're either private, irrelevant to
the demo, or too large:

- `private/` — **never** push this directory. It's in `.dockerignore` too.
- `tests/` — not used at runtime.
- `Resources/` (~27 MB raw data), `Images/` (~11 MB), `*.ipynb` — not needed.
- `.git/`, `.github/`, `.vscode/`, `.venv/`, `__pycache__/` — clutter.
- `Dockerfile` (the original one from the project root, for docker-compose)
  — you're using the HF-specific Dockerfile instead.
- `docker-compose.yml`, `Makefile`, `k8s/` — wrong deployment target.

The `.dockerignore` you copied in step 4 will exclude most of these even
if you accidentally copy them over.

---

## Step 5 — Verify the layout

Your `hf-space-high-pay` directory should look roughly like this:

```
hf-space-high-pay/
├── Dockerfile                 # HF Spaces-specific, runs both services
├── README.md                  # Space README with YAML frontmatter
├── start.sh                   # Bash entrypoint
├── .dockerignore
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
│   ├── feature_names.json
│   ├── group_means.json
│   ├── model_metrics.json
│   └── baseline_stats.json
├── Data/
│   └── cleaned_high_pay_data.csv
├── us_state/
│   └── ...
├── deploy/
│   └── huggingface/
│       └── start.sh
├── pipeline.py
├── config_schema.py
├── config.yaml
├── streamlit_app.py
├── pyproject.toml
├── requirements.txt
├── requirements-api.txt
└── LICENSE
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
git lfs track "models/*.ubj" "Data/*.csv" "us_state/*.shp" "us_state/*.dbf"
git add .gitattributes
git commit -m "Track large binaries via LFS"
git push
```

Our `models/xgb_salary_model.ubj` is 588 KB and `Data/cleaned_high_pay_data.csv`
is 1.3 MB, both well under the 10 MB warning threshold, so LFS is probably
not needed. `us_state/us_state.shp` is ~15 MB though, so if you included it
you may hit the LFS warning. You can either enable LFS (above) or just
skip the `us_state/` directory — the Geographic tab will degrade gracefully.

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
- **`models/xgb_salary_model.ubj` not found** — you forgot to copy the
  `models/` directory in Step 4. Re-copy and push again.
- **`Data/cleaned_high_pay_data.csv` not found** — same fix for `Data/`.

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

Once the Space is set up, the update flow is:

1. Make your changes in the main project
   (`Deployed_project/High_pay_Analysis_us`).
2. Copy the changed files to `hf-space-high-pay/`.
3. `git commit -am "Update X" && git push` in `hf-space-high-pay/`.
4. HF rebuilds automatically (~30 s for code-only changes, ~3 min if
   deps changed).

If you change the model, re-run `python -m scripts.train_quantile` in
the main project, then copy `models/` and `Data/` over to the Space
and push.

> **Tip**: you can also add a second git remote to the main project
> pointing at the Space repo, and use `git push space main` to sync
> only specific branches. This is faster than copying files but
> requires careful `.dockerignore` management because the Space will
> receive the full project tree including `private/` unless excluded.
> For a portfolio demo the copy-and-push flow above is safer.

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
README.md, .dockerignore, this DEPLOY.md) to the main GitHub project.
That's the deployment tooling. The Space repo itself is separate and
stays local (or gets pushed only to HF).

**Never** commit your HF access token to either repo.
