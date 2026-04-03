"""
Export the trained XGBoost model to ONNX format for faster inference.

ONNX Runtime provides 10-50x faster inference for tree models compared
to native XGBoost predict(), which matters for cost efficiency under
Kubernetes HPA auto-scaling.

Usage:
    python scripts/export_onnx.py

Output:
    models/xgb_salary_model.onnx
"""

from pathlib import Path

import numpy as np

from pipeline import FEATURES_FULL, load_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_salary_model.ubj"
ONNX_PATH = ROOT / "models" / "xgb_salary_model.onnx"


def export() -> None:
    """Convert XGBoost .ubj model to ONNX format and verify equivalence."""
    import onnxruntime as rt
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(str(MODEL_PATH))

    # Convert to ONNX
    n_features = len(FEATURES_FULL)
    initial_type = [("features", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model.get_booster(), initial_types=initial_type)

    # Save
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_PATH.stat().st_size / 1024
    print(f"ONNX model saved: {ONNX_PATH} ({size_kb:.0f} KB)")

    # Verify equivalence
    session = rt.InferenceSession(str(ONNX_PATH))
    rng = np.random.default_rng(42)
    test_input = rng.standard_normal((10, n_features)).astype(np.float32)

    xgb_preds = model.predict(test_input)
    onnx_preds = session.run(None, {"features": test_input})[0].flatten()

    max_diff = float(np.max(np.abs(xgb_preds - onnx_preds)))
    assert max_diff < 1e-4, f"ONNX output diverges: max diff = {max_diff}"
    print(f"Equivalence verified: max diff = {max_diff:.2e} (< 1e-4)")

    # Benchmark
    import time

    # XGBoost native
    single_input = test_input[:1]
    times_xgb = []
    for _ in range(500):
        s = time.perf_counter()
        model.predict(single_input)
        times_xgb.append((time.perf_counter() - s) * 1000)

    # ONNX Runtime
    times_onnx = []
    for _ in range(500):
        s = time.perf_counter()
        session.run(None, {"features": single_input})
        times_onnx.append((time.perf_counter() - s) * 1000)

    times_xgb.sort()
    times_onnx.sort()
    print("\nInference benchmark (500 runs, single prediction):")
    print(f"  XGBoost native:  p50={times_xgb[249]:.2f}ms  p99={times_xgb[494]:.2f}ms")
    print(f"  ONNX Runtime:    p50={times_onnx[249]:.2f}ms  p99={times_onnx[494]:.2f}ms")
    speedup = times_xgb[249] / times_onnx[249] if times_onnx[249] > 0 else 0
    print(f"  Speedup:         {speedup:.1f}x")


if __name__ == "__main__":
    export()
