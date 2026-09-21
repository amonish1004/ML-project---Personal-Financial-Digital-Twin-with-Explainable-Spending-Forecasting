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


def run_early_stopping_experiment():
    print("=" * 105)
    print("PERSONAL FINANCIAL DIGITAL TWIN — ISOLATED CATBOOST EARLY STOPPING EXPERIMENT")
    print("=" * 105)

    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"
    
    print("\n--- STAGE 1: LOADING & TEMPORAL PARTITIONING ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    print(f"Total Dataset Rows       : {meta['total_samples']:,}")
    print(f"TRAIN Set (2013-04..2016-12) : {meta['train_samples']:,} samples")
    print(f"VAL Set   (2017-01..2017-12) : {meta['val_samples']:,} samples")
    print(f"TEST Set  (2018-01..2018-12) : {meta['test_samples']:,} samples (HELD OUT - UNTOUCHED)")

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]

    # Verify 14 feature ordering matches expectation
    assert list(X_train.columns) == PREDICTOR_FEATURES, "Feature column ordering mismatch!"
    print(f"\nFeatures verified ({len(PREDICTOR_FEATURES)} predictors): {PREDICTOR_FEATURES}")

    # ----------------------------------------------------
    # Configuration A: Existing Baseline (300 trees, lr=0.05, depth=6)
    # ----------------------------------------------------
    print("\n--- STAGE 2: CONFIGURATION A — EXISTING BASELINE (300 iter, lr=0.05, depth=6) ---")
    model_a = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        loss_function="RMSE",
        verbose=0,
    )
    model_a.fit(X_train, y_train)

    val_preds_a = model_a.predict(X_val)
    metrics_a = calculate_regression_metrics(y_val, val_preds_a)
    raw_iter_a = model_a.get_best_iteration()
    best_iter_a_str = str(raw_iter_a) if raw_iter_a is not None else "300 (full)"

    print("  - Configuration A Trained on Train (2013-2016)")
    print(f"  - 2017 Validation Metrics: {metrics_a}")

    # ----------------------------------------------------
    # Configuration B: Experimental Early Stopping (1500 trees, lr=0.03, depth=6, early_stopping_rounds=50)
    # ----------------------------------------------------
    print("\n--- STAGE 3: CONFIGURATION B — EARLY STOPPING (1500 iter, lr=0.03, depth=6, early_stopping_rounds=50) ---")
    model_b = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        loss_function="RMSE",
        verbose=0,
    )
    model_b.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )

    val_preds_b = model_b.predict(X_val)
    metrics_b = calculate_regression_metrics(y_val, val_preds_b)
    best_iter_b = model_b.get_best_iteration()
    best_iter_b_str = str(best_iter_b) if best_iter_b is not None else "N/A"

    print(f"  - Configuration B Trained on Train (2013-2016) with Early Stopping on Val (2017)")
    print(f"  - Best Iteration: {best_iter_b}")
    print(f"  - 2017 Validation Metrics: {metrics_b}")

    # Save isolated experiment artifact in models/experiments/ (NOT production model)
    exp_artifact_path = EXPERIMENTS_DIR / "exp_catboost_early_stopping.joblib"
    joblib.dump(model_b, exp_artifact_path)
    print(f"  - Saved Isolated Experiment Artifact: {exp_artifact_path}")

    # ----------------------------------------------------
    # Comparison & Evaluation Summary
    # ----------------------------------------------------
    print("\n" + "=" * 105)
    print("COMPARISON TABLE: 2017 VALIDATION SET EVALUATION")
    print("=" * 105)
    print(f"{'Configuration':<35} | {'Iter':<6} | {'LR':<5} | {'Depth':<5} | {'Best Iter':<10} | {'MAE':<9} | {'RMSE':<9} | {'R2':<7} | {'MedAE':<9}")
    print("-" * 105)

    print(f"{'A. Existing Baseline':<35} | {300:<6} | {0.05:<5.2f} | {6:<5} | {best_iter_a_str:<10} | {metrics_a['MAE']:<9.2f} | {metrics_a['RMSE']:<9.2f} | {metrics_a['R2']:<7.4f} | {metrics_a['MedAE']:<9.2f}")
    print(f"{'B. Early Stopping (Exp)':<35} | {1500:<6} | {0.03:<5.2f} | {6:<5} | {best_iter_b_str:<10} | {metrics_b['MAE']:<9.2f} | {metrics_b['RMSE']:<9.2f} | {metrics_b['R2']:<7.4f} | {metrics_b['MedAE']:<9.2f}")
    print("-" * 105)

    r2_diff = metrics_b['R2'] - metrics_a['R2']
    mae_diff = metrics_b['MAE'] - metrics_a['MAE']
    rmse_diff = metrics_b['RMSE'] - metrics_a['RMSE']

    print("\n--- VALIDATION IMPACT ASSESSMENT ---")
    if r2_diff > 0 and mae_diff < 0:
        print(f"  -> RESULT: Configuration B (Early Stopping) IMPROVED 2017 Validation performance!")
        print(f"     * R² Delta   : {r2_diff:+.4f} (Increased from {metrics_a['R2']} to {metrics_b['R2']})")
        print(f"     * MAE Delta  : {mae_diff:+.2f} (Decreased from {metrics_a['MAE']} to {metrics_b['MAE']})")
        print(f"     * RMSE Delta : {rmse_diff:+.2f} (Decreased from {metrics_a['RMSE']} to {metrics_b['RMSE']})")
    elif r2_diff > 0:
        print(f"  -> RESULT: Configuration B (Early Stopping) IMPROVED 2017 Validation R²!")
        print(f"     * R² Delta   : {r2_diff:+.4f} (Increased from {metrics_a['R2']} to {metrics_b['R2']})")
        print(f"     * MAE Delta  : {mae_diff:+.2f} (Changed from {metrics_a['MAE']} to {metrics_b['MAE']})")
        print(f"     * RMSE Delta : {rmse_diff:+.2f} (Decreased from {metrics_a['RMSE']} to {metrics_b['RMSE']})")
    else:
        print(f"  -> RESULT: Mixed impact. R² Delta = {r2_diff:+.4f}, MAE Delta = {mae_diff:+.2f}.")

    print("\n" + "=" * 105)
    print("EXPERIMENT COMPLETED — PRODUCTION MODEL models/catboost_model.joblib WAS UNTOUCHED.")
    print("=" * 105)


if __name__ == "__main__":
    run_early_stopping_experiment()
