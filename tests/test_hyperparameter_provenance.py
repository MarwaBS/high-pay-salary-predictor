"""Every tuned hyper-parameter in ``config.yaml`` must trace to the study that chose it.

Binds ``config.yaml``, ``scripts/tune.py`` and ``models/tuning_study.json``
together, so editing a shipped value without re-running the study fails rather
than passing quietly.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.tune as tune
from pipeline import train_test_positions
from scripts.tune import SEARCH_SPACE, TUNED_KEYS, _sample, cv_pinball, training_frame

REPO_ROOT = Path(__file__).parent.parent
STUDY_PATH = REPO_ROOT / "models" / "tuning_study.json"

#: How far a re-run may land from a recorded score, XGBoost's float reductions
#: differing by build.
REPRODUCTION_TOLERANCE = 0.01

#: Any percentage a doc publishes as this tolerance, wherever it sits.
PUBLISHED_TOLERANCE = re.compile(r"to within ([0-9.]+)%|([0-9.]+)% relative tolerance")
MINIMUM_PUBLISHED = 2

#: A doc may not claim exact reproduction without scoping it to one machine or
#: to the tolerance.
EXACTNESS = re.compile(r"bit-exact|bit-identical|byte-identical|identical metrics|identical artefacts", re.I)
RERUN = re.compile(r"re-?run|re-?running|regenerat|reproduc|retrain", re.I)
SCOPED = re.compile(r"on one machine|same machine|to within [0-9.]+%", re.I)
CFG = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))["model"]


@pytest.fixture(scope="module")
def tune_inputs() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    return {
        "edu_order": cfg["education_order"],
        "region_map": {s: r for r, states in cfg["regions"].items() for s in states},
    }


@pytest.fixture(scope="module")
def tiny_train() -> pd.DataFrame:
    """A deterministic slice — the search mechanics do not need the full frame."""
    raw = pd.read_csv(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")
    return raw.sample(n=600, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def study() -> dict:
    assert STUDY_PATH.exists(), "no tuning study committed — the shipped values have no producer"
    return json.loads(STUDY_PATH.read_text(encoding="utf-8"))


def _doc_surface() -> list[Path]:
    """The published text files, from git rather than a hand-kept directory list."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.md", "*.yml", "*.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    paths = [REPO_ROOT / name for name in listed.split("\0") if name]
    assert len(paths) >= 10, f"only {len(paths)} tracked docs found — this scan is looking at nothing"
    return paths


def test_every_published_reproduction_tolerance_is_the_enforced_one():
    """No doc may print a tolerance other than the one the re-derivations assert."""
    published = [
        (path.name, number, figure)
        for path in _doc_surface()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for match in PUBLISHED_TOLERANCE.finditer(line)
        for figure in [next(g for g in match.groups() if g)]
    ]
    assert len(published) >= MINIMUM_PUBLISHED, (
        f"only {len(published)} published tolerances found; a rewording leaves this scanning nothing"
    )
    wrong = [
        f"{name}:{number} prints {figure}%"
        for name, number, figure in published
        if float(figure) / 100 != pytest.approx(REPRODUCTION_TOLERANCE)
    ]
    assert not wrong, f"{REPRODUCTION_TOLERANCE:.0%} is enforced but " + "; ".join(wrong)


def test_no_doc_claims_exact_reproduction_unscoped():
    """An exactness claim about a re-run has to name one machine or the tolerance."""
    offenders = []
    for path in _doc_surface():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if EXACTNESS.search(line) and RERUN.search(line) and not SCOPED.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()[:100]}")
    assert not offenders, "unscoped exact-reproduction claims:\n" + "\n".join(offenders)


@pytest.mark.parametrize("key", TUNED_KEYS)
def test_shipped_value_is_the_one_the_study_retained(study, key):
    """The study retained the incumbent, so config must still equal it.

    If a future study wins, this is the test that forces the study record and
    the shipped config to move together.
    """
    retained = study["incumbent"] if study["improvement_vs_incumbent"] <= 0 else study["best"]
    assert CFG[key] == retained["params"][key], (
        f"config.yaml ships {key}={CFG[key]!r} but the study retained {retained['params'][key]!r}"
    )


@pytest.mark.parametrize("key", TUNED_KEYS)
def test_config_matches_the_artefact_that_was_actually_trained(key):
    """``model_metrics.json`` records what the shipped model was fitted with.

    Config and the study agreeing is not enough: both are editable text, so a
    value neither run ever used could sit in both. The artefact is the only
    record produced by training, which makes it the tie-breaker.
    """
    trained = json.loads((REPO_ROOT / "models" / "model_metrics.json").read_text(encoding="utf-8"))
    assert CFG[key] == trained["hyperparameters"][key], (
        f"config ships {key}={CFG[key]!r} but the trained artefact records {trained['hyperparameters'][key]!r}"
    )


@pytest.mark.parametrize("key", TUNED_KEYS)
def test_shipped_value_lies_inside_the_searched_space(key):
    """A value outside the space was never a candidate, so the study cannot vouch for it."""
    kind, low, high = SEARCH_SPACE[key]
    assert low <= CFG[key] <= high, f"{key}={CFG[key]} is outside the searched range [{low}, {high}]"


def test_study_was_run_under_the_shipped_reproducibility_settings(study):
    assert study["seed"] == CFG["random_state"]
    assert study["cv_folds"] == CFG["cv_folds"]
    assert study["n_jobs"] == CFG["n_jobs"], "a multi-threaded study is not reproducible"


def test_the_search_actually_discriminated(study):
    """A space where every candidate scores alike would vouch for nothing.

    Without this, a study that sampled a single degenerate region could 'confirm'
    any incumbent at all.
    """
    losses = [trial["cv_pinball"] for trial in study["all_trials"]]
    assert len(losses) >= 30, f"only {len(losses)} trials — too few to conclude anything"
    assert max(losses) - min(losses) > 100, "search space produced no meaningful spread in loss"


def test_every_tuned_key_is_recorded_in_the_study(study):
    assert set(TUNED_KEYS) == set(study["incumbent"]["params"]), "study and search space disagree on the tuned set"


class TestTheSearchItself:
    """The producer has to work, not merely exist.

    A sampler that drifted outside its stated space, or a scorer that varied
    run to run, would make every number in the study record unreproducible.
    """

    def test_sampling_is_seed_deterministic_and_inside_the_space(self):
        first = [_sample(np.random.default_rng(42)) for _ in range(20)]
        second = [_sample(np.random.default_rng(42)) for _ in range(20)]
        assert first == second, "same seed produced a different search"

        for candidate in first:
            assert set(candidate) == set(TUNED_KEYS)
            for key, value in candidate.items():
                kind, low, high = SEARCH_SPACE[key]
                assert low <= value <= high, f"{key}={value} escaped [{low}, {high}]"
                assert isinstance(value, int) if kind == "int" else isinstance(value, float)

    def test_scoring_a_candidate_is_reproducible(self, tiny_train, tune_inputs):
        """Two runs of the scorer on identical inputs must agree exactly."""
        params = {"n_estimators": 12, "max_depth": 2, "learning_rate": 0.2, "reg_lambda": 1.0}
        scores = [cv_pinball(tiny_train, params, folds=3, seed=42, n_jobs=1, **tune_inputs) for _ in range(2)]
        assert scores[0] == scores[1], f"scorer is not deterministic: {scores}"
        assert scores[0] > 0, "pinball loss must be positive"

    def test_the_scorer_separates_different_candidates(self, tiny_train, tune_inputs):
        """A scorer that ranked everything alike would vouch for any incumbent.

        Only the magnitude is asserted, not the sign: on a small slice a single
        stump can beat a 60-tree model, so which of these two wins is a property
        of the sample size rather than of the scorer.
        """
        common = dict(folds=3, seed=42, n_jobs=1, **tune_inputs)
        deep = cv_pinball(tiny_train, {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1}, **common)
        stump = cv_pinball(tiny_train, {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.001}, **common)
        assert abs(deep - stump) > 100, f"scorer barely separated two very different models: {deep} vs {stump}"


class TestTheStudyRecordIsSelfConsistent:
    """A record whose numbers disagree with each other vouches for nothing.

    Each assertion here is arithmetic on the file alone, so an edit that leaves
    the record internally coherent passes them. Re-running a score is what a
    fabricated one cannot survive; that is the two re-derivation tests below.
    """

    def test_improvement_equals_the_difference_it_reports(self, study):
        stated = study["improvement_vs_incumbent"]
        derived = study["incumbent"]["cv_pinball"] - study["best"]["cv_pinball"]
        assert stated == pytest.approx(derived, abs=1e-6), f"study reports {stated}, its own scores give {derived}"

    def test_trial_count_matches_the_trials_recorded(self, study):
        assert study["trials"] == len(study["all_trials"])

    def test_best_is_the_argmin_of_the_recorded_trials(self, study):
        winner = min(study["all_trials"], key=lambda trial: trial["cv_pinball"])
        assert study["best"]["trial"] == winner["trial"]
        assert study["best"]["cv_pinball"] == winner["cv_pinball"]
        assert study["best"]["params"] == winner["params"]

    def test_every_trial_searched_the_declared_space(self, study):
        for trial in study["all_trials"]:
            assert set(trial["params"]) == set(TUNED_KEYS), f"trial {trial['trial']} tuned a different set"
            for key, value in trial["params"].items():
                low, high = SEARCH_SPACE[key][1:]
                assert low <= value <= high, f"trial {trial['trial']} {key}={value} outside [{low}, {high}]"

    def test_trials_actually_differ_from_one_another(self, study):
        """Identical candidates scored repeatedly would fake a spread."""
        seen = {tuple(sorted(trial["params"].items())) for trial in study["all_trials"]}
        assert len(seen) == len(study["all_trials"]), "the study repeated candidates"

    def test_study_names_the_dataset_it_scored(self, study):
        """Ties the record to the CSV; a study run on other data cannot vouch for these values."""
        from scripts.train_quantile import _hash_training_data

        assert study["data_sha256"] == _hash_training_data(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")


class TestSelectionNeverSeesHeldOutData:
    """The one property the whole study rests on.

    If the scorer sees the test split, or encodes a validation fold with means
    derived from its own rows, every metric downstream of these values is
    optimistic and nothing else in the suite would notice.
    """

    def test_the_tuning_frame_is_exactly_the_trainer_train_split(self):
        raw = pd.read_csv(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")
        train_pos, test_pos = train_test_positions(
            len(raw), test_size=CFG["test_size"], random_state=CFG["random_state"]
        )
        tuned_on = training_frame(raw, test_size=CFG["test_size"], seed=CFG["random_state"])

        assert len(tuned_on) == len(train_pos)
        assert tuned_on.equals(raw.iloc[train_pos].reset_index(drop=True)), "tuning frame is not the train split"
        held_out = raw.iloc[test_pos]
        assert not set(map(tuple, tuned_on.to_numpy())) & set(map(tuple, held_out.to_numpy())), (
            "a held-out row reached the tuning frame"
        )

    def test_target_encodings_are_rederived_per_fold(self, tiny_train, tune_inputs, monkeypatch):
        """Means computed once over the whole frame would leak each fold's own targets."""
        sizes: list[int] = []
        real = tune.compute_group_means

        def _spy(frame):
            sizes.append(len(frame))
            return real(frame)

        monkeypatch.setattr(tune, "compute_group_means", _spy)
        cv_pinball(tiny_train, {"n_estimators": 5, "max_depth": 2}, folds=3, seed=42, n_jobs=1, **tune_inputs)

        assert len(sizes) == 3, f"expected one encoding per fold, got {len(sizes)}"
        assert all(size < len(tiny_train) for size in sizes), f"a fold encoded from the whole frame: {sizes}"


def test_a_real_run_records_the_argmin_as_best(tmp_path):
    """Drives the entry point, because the committed study cannot catch a
    selection bug: a run that picked the worst candidate would still write a
    self-consistent file. Two folds and three trials keep it quick."""
    cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["cv_folds"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "study.json"

    result = subprocess.run(
        [sys.executable, "-m", "scripts.tune", "--config", str(config_path), "--trials", "3", "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CI": "true"},
    )
    assert result.returncode == 0, result.stderr[-2000:]

    produced = json.loads(out.read_text(encoding="utf-8"))
    winner = min(produced["all_trials"], key=lambda trial: trial["cv_pinball"])
    assert produced["best"]["trial"] == winner["trial"], "the run did not select its own argmin"
    assert produced["improvement_vs_incumbent"] == pytest.approx(
        produced["incumbent"]["cv_pinball"] - produced["best"]["cv_pinball"], abs=1e-6
    )


def test_the_recorded_incumbent_score_re_derives(study, tune_inputs):
    """Recompute the incumbent's score instead of trusting the file.

    Arithmetic consistency is not enough: a record can be internally coherent
    and still report a number no run produced. Re-running the CV is the only
    check a forged score cannot satisfy, so it pays the ~4s.

    The tolerance is relative, not exact. XGBoost's float reductions differ by
    build: the same parameters and seed give 17524.88 on the machine that
    recorded the study, 17541.41 under Linux/CPython 3.11 and 17544.54 under
    3.12 — a 0.11% spread. Asserting equality would be asserting a portability
    the repo does not have; 1% still separates a real run from a fabricated one
    by orders of magnitude.
    """
    raw = pd.read_csv(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")
    train = training_frame(raw, test_size=CFG["test_size"], seed=study["seed"])
    recomputed = cv_pinball(
        train,
        study["incumbent"]["params"],
        seed=study["seed"],
        folds=study["cv_folds"],
        n_jobs=study["n_jobs"],
        **tune_inputs,
    )
    assert recomputed == pytest.approx(study["incumbent"]["cv_pinball"], rel=REPRODUCTION_TOLERANCE), (
        f"study records {study['incumbent']['cv_pinball']}, re-running gives {recomputed} — "
        "further apart than build-to-build float variation explains"
    )


def test_the_recorded_best_score_re_derives(study, tune_inputs):
    """The winner's score is re-run too, because the conclusion compares the pair.

    Re-deriving the incumbent alone leaves the other half of the comparison taken
    on trust, so a record could keep a true incumbent beside an invented winner
    and stay coherent. The 58 remaining trial scores are not re-run: they cost a
    full CV each and no conclusion rests on them individually.
    """
    raw = pd.read_csv(REPO_ROOT / "Data" / "cleaned_high_pay_data.csv")
    train = training_frame(raw, test_size=CFG["test_size"], seed=study["seed"])
    recomputed = cv_pinball(
        train,
        study["best"]["params"],
        seed=study["seed"],
        folds=study["cv_folds"],
        n_jobs=study["n_jobs"],
        **tune_inputs,
    )
    assert recomputed == pytest.approx(study["best"]["cv_pinball"], rel=REPRODUCTION_TOLERANCE), (
        f"study records {study['best']['cv_pinball']} for trial {study['best']['trial']}, "
        f"re-running its parameters gives {recomputed}"
    )
