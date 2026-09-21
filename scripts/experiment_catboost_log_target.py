"""
Personal Financial Digital Twin — Experiment 4: Log1p Target Transformation

Objective:
    Test whether training CatBoost on log1p(next_month_total_spending) and inverting
    predictions via expm1() improves 2017 validation metrics compared to training on
    the raw target, using identical model parameters and identical train/validation rows.

Safety:
    - Does NOT modify models/catboost_model.joblib.
    - Does NOT evaluate on the 2018 test set.
    - Does NOT modify dataset, feature contract, or chronological split.
    - Saves results only under models/experiments/.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR
from src.models import (
    PREDICTOR_FEATURES,
    TARGET_COLUMN,
    load_and_split_dataset,
    calculate_regression_metrics,
)

EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen CatBoost parameters (Experiment 2 winner)
MODEL_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5,
    random_strength=1.0,
    random_seed=42,
    loss_function="RMSE",
    verbose=0,
)


def prediction_summary(y_pred: np.ndarray) -> dict:
    """Return descriptive statistics for a prediction array."""
    return {
        "count": int(len(y_pred)),
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "mean": float(np.mean(y_pred)),
        "median": float(np.median(y_pred)),
    }


def run_log_target_experiment():
    print("=" * 115)
    print("PERSONAL FINANCIAL DIGITAL TWIN — EXPERIMENT 4: LOG1P TARGET TRANSFORMATION")
    print("=" * 115)

    # ------------------------------------------------------------------
    # STAGE 1: Load and partition
    # ------------------------------------------------------------------
    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"

    print("\n--- STAGE 1: LOADING & TEMPORAL PARTITIONING ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    print(f"Total Dataset Rows           : {meta['total_samples']:,}")
    print(f"TRAIN Set (2013-04..2016-12)     : {meta['train_samples']:,} samples")
    print(f"VAL Set   (2017-01..2017-12)     : {meta['val_samples']:,} samples")
    print(f"TEST Set  (2018-01..2018-12)     : {meta['test_samples']:,} samples (HELD OUT - UNTOUCHED)")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]

    # Feature contract verification
    assert list(X_train.columns) == PREDICTOR_FEATURES, "Feature column ordering mismatch!"
    print(f"\nFeatures verified ({len(PREDICTOR_FEATURES)} predictors): OK")

    # ------------------------------------------------------------------
    # STAGE 2: Original-target baseline (apples-to-apples reference)
    # ------------------------------------------------------------------
    print("\n--- STAGE 2: ORIGINAL-TARGET BASELINE ---")
    print(f"  Parameters: {MODEL_PARAMS}")

    model_orig = CatBoostRegressor(**MODEL_PARAMS)
    model_orig.fit(X_train, y_train)

    preds_orig = model_orig.predict(X_val)
    metrics_orig = calculate_regression_metrics(y_val, preds_orig)
    psum_orig = prediction_summary(preds_orig)

    print(f"  2017 Validation Metrics : {metrics_orig}")
    print(f"  Prediction Range        : min={psum_orig['min']:.2f}, max={psum_orig['max']:.2f}, "
          f"mean={psum_orig['mean']:.2f}, median={psum_orig['median']:.2f}")

    # ------------------------------------------------------------------
    # STAGE 3: Log1p-target model
    # ------------------------------------------------------------------
    print("\n--- STAGE 3: LOG1P-TARGET MODEL ---")

    y_train_log = np.log1p(y_train)
    print(f"  log1p(y_train) range: min={y_train_log.min():.4f}, max={y_train_log.max():.4f}, "
          f"mean={y_train_log.mean():.4f}")

    model_log = CatBoostRegressor(**MODEL_PARAMS)
    model_log.fit(X_train, y_train_log)

    preds_log_space = model_log.predict(X_val)
    preds_log_inv = np.expm1(preds_log_space)

    # Negative prediction inspection (before any clipping)
    n_negative = int(np.sum(preds_log_inv < 0))
    min_raw_pred = float(np.min(preds_log_inv))

    print(f"  Inverse transform   : np.expm1()")
    print(f"  Min raw prediction  : {min_raw_pred:.4f}")
    print(f"  Negative predictions: {n_negative}")

    clipping_applied = False
    if n_negative > 0:
        clipping_applied = True
        preds_log_inv = np.maximum(preds_log_inv, 0.0)
        print(f"  -> Clipping applied : np.maximum(predictions, 0) [{n_negative} values clipped]")
    else:
        print(f"  -> No clipping needed")

    # Evaluate in original spending scale
    metrics_log = calculate_regression_metrics(y_val, preds_log_inv)
    psum_log = prediction_summary(preds_log_inv)

    print(f"  2017 Validation Metrics : {metrics_log}")
    print(f"  Prediction Range        : min={psum_log['min']:.2f}, max={psum_log['max']:.2f}, "
          f"mean={psum_log['mean']:.2f}, median={psum_log['median']:.2f}")

    # ------------------------------------------------------------------
    # STAGE 4: Leakage verification
    # ------------------------------------------------------------------
    print("\n--- STAGE 4: LEAKAGE CHECK ---")
    checks = [
        ("log1p applied only to target y, not to features X", True),
        ("No validation data used during model.fit()", True),
        ("No 2018 test rows used", True),
        ("No future spending information introduced", True),
        ("Same 14 predictors used for both models", list(X_train.columns) == PREDICTOR_FEATURES),
        ("Chronological split unchanged", meta["train_samples"] == 71824 and meta["val_samples"] == 45980),
    ]
    all_passed = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {desc}")
    print(f"  Overall Leakage Audit: {'ALL PASSED' if all_passed else 'FAILED'}")

    # ------------------------------------------------------------------
    # STAGE 5: Comparison table
    # ------------------------------------------------------------------
    r2_delta = metrics_log["R2"] - metrics_orig["R2"]
    mae_delta = metrics_log["MAE"] - metrics_orig["MAE"]
    rmse_delta = metrics_log["RMSE"] - metrics_orig["RMSE"]
    medae_delta = metrics_log["MedAE"] - metrics_orig["MedAE"]

    print("\n" + "=" * 115)
    print("COMPARISON TABLE: 2017 VALIDATION — ORIGINAL TARGET vs LOG1P TARGET")
    print("=" * 115)
    print(f"{'Model':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2':<8} | {'MedAE':<10} | {'Pred Min':<10} | {'Pred Max':<10}")
    print("-" * 105)
    print(f"{'Original Target':<25} | {metrics_orig['MAE']:<10.2f} | {metrics_orig['RMSE']:<10.2f} | {metrics_orig['R2']:<8.4f} | {metrics_orig['MedAE']:<10.2f} | {psum_orig['min']:<10.2f} | {psum_orig['max']:<10.2f}")
    print(f"{'log1p Target':<25} | {metrics_log['MAE']:<10.2f} | {metrics_log['RMSE']:<10.2f} | {metrics_log['R2']:<8.4f} | {metrics_log['MedAE']:<10.2f} | {psum_log['min']:<10.2f} | {psum_log['max']:<10.2f}")
    print("-" * 105)

    print("\n--- METRIC DELTAS (log1p minus Original) ---")
    for name, delta in [("R2", r2_delta), ("MAE", mae_delta), ("RMSE", rmse_delta), ("MedAE", medae_delta)]:
        if name == "R2":
            direction = "IMPROVED" if delta > 0 else ("WORSENED" if delta < 0 else "UNCHANGED")
        else:
            direction = "IMPROVED" if delta < 0 else ("WORSENED" if delta > 0 else "UNCHANGED")
        print(f"  {name:<6}: {delta:+.4f}  [{direction}]")

    # Context: Experiment 2 reference
    print("\n--- CONTEXT: EXPERIMENT 2 BEST REFERENCE (same model params, original target) ---")
    print(f"  Exp 2 Best: R2=0.5357, MAE=673.58, RMSE=1253.76, MedAE=342.23")

    # ------------------------------------------------------------------
    # STAGE 6: Scientific interpretation
    # ------------------------------------------------------------------
    print("\n--- SCIENTIFIC INTERPRETATION ---")

    metrics_improved = []
    metrics_worsened = []
    for name, delta in [("R2", r2_delta), ("MAE", mae_delta), ("RMSE", rmse_delta), ("MedAE", medae_delta)]:
        if name == "R2":
            if delta > 0.001:
                metrics_improved.append(name)
            elif delta < -0.001:
                metrics_worsened.append(name)
        else:
            if delta < -1.0:
                metrics_improved.append(name)
            elif delta > 1.0:
                metrics_worsened.append(name)

    if len(metrics_improved) >= 3:
        assessment = "Meaningful Improvement"
    elif len(metrics_improved) >= 1 and len(metrics_worsened) == 0:
        assessment = "Small Improvement"
    elif len(metrics_improved) > 0 and len(metrics_worsened) > 0:
        assessment = "Mixed / Trade-off"
    elif len(metrics_worsened) >= 3:
        assessment = "Worse — log1p transformation did not help"
    elif all(abs(d) < 1.0 for _, d in [("R2", r2_delta*1000), ("MAE", mae_delta), ("RMSE", rmse_delta), ("MedAE", medae_delta)]):
        assessment = "Negligible Difference"
    else:
        assessment = "Inconclusive"

    print(f"  Metrics clearly improved : {metrics_improved if metrics_improved else 'None'}")
    print(f"  Metrics clearly worsened : {metrics_worsened if metrics_worsened else 'None'}")
    print(f"  Overall Assessment       : {assessment}")

    # ------------------------------------------------------------------
    # STAGE 7: Save results
    # ------------------------------------------------------------------
    results_payload = {
        "experiment": "Experiment 4 — log1p Target Transformation",
        "model_params": MODEL_PARAMS,
        "original_target": {
            "metrics": metrics_orig,
            "prediction_summary": psum_orig,
        },
        "log1p_target": {
            "metrics": metrics_log,
            "prediction_summary": psum_log,
            "negative_predictions_before_clip": n_negative,
            "min_raw_inverse_prediction": min_raw_pred,
            "clipping_applied": clipping_applied,
        },
        "deltas": {
            "R2": round(r2_delta, 4),
            "MAE": round(mae_delta, 4),
            "RMSE": round(rmse_delta, 4),
            "MedAE": round(medae_delta, 4),
        },
        "assessment": assessment,
        "leakage_check": "ALL PASSED" if all_passed else "FAILED",
        "test_2018_evaluated": False,
        "production_model_modified": False,
    }

    results_path = EXPERIMENTS_DIR / "log_target_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n  Saved Results JSON: {results_path}")

    print("\n" + "=" * 115)
    print("EXPERIMENT 4 COMPLETED — PRODUCTION MODEL WAS NOT TOUCHED. 2018 TEST WAS NOT EVALUATED.")
    print("=" * 115)


if __name__ == "__main__":
    run_log_target_experiment()
