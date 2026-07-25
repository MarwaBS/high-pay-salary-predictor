"""
Regression guard: exactly ONE trainer under ``scripts/`` may write the
production ``models/xgb_salary_model.ubj`` artefact.

Two trainers writing the same output path is the failure this guards
against: a point-estimate trainer (``reg:squarederror``) sharing the path
with the multi-quantile production trainer lets ``make hpo`` silently
clobber the production model with a point estimator, degrading the API to
``(p, p, p)`` degenerate intervals. This test fails if any sibling trainer
writing that path appears.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _python_modules_in_scripts() -> list[Path]:
    """Return top-level .py files under scripts/, excluding __init__."""
    return sorted(p for p in SCRIPTS.glob("*.py") if p.name != "__init__.py")


def _writes_model_artifact(module: Path) -> bool:
    """A "trainer" writes the production model artefact via ``save_model()``.

    Non-trainer utilities under ``scripts/`` (e.g. the artefact integrity
    verifier, which only reads and hashes) are not part of the dual-trainer footgun.
    """
    return "save_model(" in module.read_text(encoding="utf-8")


def test_exactly_one_trainer_module_exists():
    """Only ``scripts/train_quantile.py`` may WRITE the production model artefact.

    The footgun is two *trainers* clobbering the same output path — not two
    files under ``scripts/``. Non-trainer utilities are allowed; a new trainer
    must come with an explicit update to this test.
    """
    trainers = [m.name for m in _python_modules_in_scripts() if _writes_model_artifact(m)]
    assert trainers == ["train_quantile.py"], (
        f"Expected exactly one trainer (writing the model artefact) under scripts/ "
        f"('train_quantile.py'), found: {trainers}. Shipping multiple trainers that "
        f"write to the same artefact path is the dual-trainer footgun."
    )


def test_quantile_trainer_uses_quantile_objective():
    """The remaining trainer must use ``reg:quantileerror``.

    Parses the trainer source as an AST and asserts a literal string
    ``"reg:quantileerror"`` appears in it. Guards against a future
    accidental regression that swaps the objective back to squared error.
    """
    trainer = SCRIPTS / "train_quantile.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "reg:quantileerror" in literals, (
        "scripts/train_quantile.py must train with objective='reg:quantileerror'. "
        "Falling back to 'reg:squarederror' would re-introduce the v1 "
        "point-estimate framing that the v2 quantile reframe replaced."
    )


def test_no_mlflow_or_optuna_imports_in_scripts():
    """No module under ``scripts/`` may import MLflow or Optuna.

    The lean trainer deliberately avoids an experiment-tracking stack; a
    reappearance means either a heavyweight trainer was reintroduced or
    something else is pulling that stack in.
    """
    forbidden = {"mlflow", "optuna"}
    for module in _python_modules_in_scripts():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden, (
                        f"{module.name} imports '{root}' — scripts/ must not depend on an experiment-tracking stack."
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden, (
                    f"{module.name} imports from '{root}' — scripts/ must not depend on an experiment-tracking stack."
                )
