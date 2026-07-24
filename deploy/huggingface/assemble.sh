#!/usr/bin/env bash
# Assemble the HF Space runtime snapshot from a checkout of this repo.
#
#   assemble.sh <repo_dir> <space_dir>
#
# Overlays onto <space_dir> (a clone of the Space repo) EXACTLY the files the
# Space's Docker build needs (see deploy/huggingface/Dockerfile's COPY list),
# and nothing else. Runtime code dirs are removed-then-copied so a file deleted
# from the repo is deleted from the Space too.
#
# WHY THIS EXISTS: the Space was hand-deployed once (deploy/huggingface/DEPLOY.md)
# and never re-pushed — it served a 3-month-stale April revision behind a "Live
# Demo" badge. A demo only a human remembers to redeploy always rots; deploys
# must ride main. This script is the SINGLE definition of "what the Space should
# contain", used by BOTH the deploy job (overlay -> commit -> push) and the
# weekly drift guard (overlay -> any diff means the Space is stale -> fail), so
# the two can never disagree.
set -euo pipefail

REPO_DIR="${1:?usage: assemble.sh <repo_dir> <space_dir>}"
SPACE_DIR="${2:?usage: assemble.sh <repo_dir> <space_dir>}"

# ── Space-specific files (Docker build entrypoints + HF card) ────────────────
# The Dockerfile + README live at the Space repo ROOT (HF convention: HF builds
# the root Dockerfile and renders the root README as the card). start.sh is
# copied to deploy/huggingface/start.sh because the Dockerfile references it by
# THAT path (`COPY deploy/huggingface/start.sh …`) — putting it at the root
# would make the Space's `docker build` fail on a missing COPY source.
# tests/test_space_snapshot.py enforces that assemble.sh satisfies every COPY
# source the Dockerfile needs.
cp "$REPO_DIR/deploy/huggingface/Dockerfile" "$SPACE_DIR/Dockerfile"
cp "$REPO_DIR/deploy/huggingface/README.md" "$SPACE_DIR/README.md"
mkdir -p "$SPACE_DIR/deploy/huggingface"
cp "$REPO_DIR/deploy/huggingface/start.sh" "$SPACE_DIR/deploy/huggingface/start.sh"

# ── Runtime dependency sets (both files the image's pip installs consume) ────
cp "$REPO_DIR/requirements-api.txt" "$SPACE_DIR/requirements-api.txt"
cp "$REPO_DIR/deploy/huggingface/requirements-space.txt" "$SPACE_DIR/deploy/huggingface/requirements-space.txt"

# ── Runtime code (fresh copy so deletions propagate) ─────────────────────────
rm -rf "$SPACE_DIR/api" "$SPACE_DIR/scripts"
cp -r "$REPO_DIR/api" "$SPACE_DIR/api"
cp -r "$REPO_DIR/scripts" "$SPACE_DIR/scripts"
cp "$REPO_DIR/pipeline.py" "$SPACE_DIR/pipeline.py"
cp "$REPO_DIR/config_schema.py" "$SPACE_DIR/config_schema.py"
cp "$REPO_DIR/config.yaml" "$SPACE_DIR/config.yaml"
cp "$REPO_DIR/streamlit_app.py" "$SPACE_DIR/streamlit_app.py"
cp "$REPO_DIR/pyproject.toml" "$SPACE_DIR/pyproject.toml"

# ── Serving artefacts: sync the committed model set wholesale ────────────────
# Ships every artefact the API loads (the same set tests/test_model_registry.py
# pins). Removing first drops any stale artefact the old hand-deploy left behind.
rm -rf "$SPACE_DIR/models"
mkdir -p "$SPACE_DIR/models"
cp "$REPO_DIR"/models/*.ubj "$SPACE_DIR/models/"
cp "$REPO_DIR"/models/*.json "$SPACE_DIR/models/"

# ── Dataset baked into the image (no volumes on HF Spaces) ───────────────────
mkdir -p "$SPACE_DIR/Data"
cp "$REPO_DIR/Data/cleaned_high_pay_data.csv" "$SPACE_DIR/Data/cleaned_high_pay_data.csv"

# ── Never ship bytecode caches (the hand-assembled Space did) ────────────────
find "$SPACE_DIR" -type d -name "__pycache__" -not -path "$SPACE_DIR/.git/*" -exec rm -rf {} +

echo "Snapshot assembled into $SPACE_DIR"
