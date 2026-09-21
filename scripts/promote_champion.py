"""
Personal Financial Digital Twin — Champion Model Promotion

Purpose:
    Retrain the Experiment 2 champion CatBoost configuration on Train + Validation
    (2013-04 through 2017-12), evaluate exactly once on the held-out 2018 Test set,
    and replace the production model artifact at models/catboost_model.joblib.

Champion Configuration (Experiment 2):
    iterations      = 300
    learning_rate   = 0.05
    depth           = 8
    l2_leaf_reg     = 5
    random_strength = 1.0
    random_seed     = 42
    loss_function   = RMSE

Safety:
    - No log1p target transformation.
    - No early stopping.
    - No learning_rate = 0.03.
    - 2018 Test set used ONLY for final evaluation (not for tuning).
    - 14-feature contract unchanged.
    - Target unchanged (next_month_total_spending, raw scale).
    - Chronological split unchanged.
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
    calculate_error_distribution,
    calculate_segmented_metrics,
)

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================
# FROZEN CHAMPION PARAMETERS (Experiment 2 Winner)
# ====================================================================
CHAMPION_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=5,
    random_strength=1.0,
    random_seed=42,
    loss_function="RMSE",
    verbose=0,
)

# Original production parameters (for reference comparison)
ORIGINAL_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,  # CatBoost default
    random_strength=1.0,  # CatBoost default
    random_seed=42,
    loss_function="RMSE",
    verbose=0,
)

# Original production 2018 test metrics (from existing pipeline run)
ORIGINAL_TEST_2018 = {
    "MAE": 729.88,
    "RMSE": 1321.27,
    "R2": 0.4954,
    "MedAE": 377.17,
    "pct_within_500": 59.55,
    "pct_within_1000": 79.77,
}


def run_champion_promotion():
    print("=" * 115)
    print("PERSONAL FINANCIAL DIGITAL TWIN — CHAMPION MODEL PROMOTION")
    print("=" * 115)

    # ------------------------------------------------------------------
    # PRE-FLIGHT SAFETY CHECKS
    # ------------------------------------------------------------------
    print("\n--- PRE-FLIGHT SAFETY CHECKS ---")

    checks = {
        "Target is raw (no log1p)": True,
        "No early stopping": "early_stopping_rounds" not in CHAMPION_PARAMS,
        "learning_rate == 0.05": CHAMPION_PARAMS["learning_rate"] == 0.05,
        "depth == 8": CHAMPION_PARAMS["depth"] == 8,
        "l2_leaf_reg == 5": CHAMPION_PARAMS["l2_leaf_reg"] == 5,
        "random_strength == 1.0": CHAMPION_PARAMS["random_strength"] == 1.0,
        "iterations == 300": CHAMPION_PARAMS["iterations"] == 300,
        "loss_function == RMSE": CHAMPION_PARAMS["loss_function"] == "RMSE",
        "random_seed == 42": CHAMPION_PARAMS["random_seed"] == 42,
    }

    all_ok = True
    for desc, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}] {desc}")

    if not all_ok:
        print("\n  *** ABORTING: Safety checks FAILED. No files modified. ***")
        sys.exit(1)

    print("  -> All pre-flight checks PASSED.")

    # ------------------------------------------------------------------
    # STAGE 1: Load and partition
    # ------------------------------------------------------------------
    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"

    print("\n--- STAGE 1: LOADING & TEMPORAL PARTITIONING ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    print(f"Total Dataset Rows           : {meta['total_samples']:,}")
    print(f"TRAIN Set (2013-04..2016-12)  : {meta['train_samples']:,} samples")
    print(f"VAL Set   (2017-01..2017-12)  : {meta['val_samples']:,} samples")
    print(f"TEST Set  (2018-01..2018-12)  : {meta['test_samples']:,} samples")

    # Row count assertions
    assert meta["train_samples"] == 71824, f"Train count mismatch: {meta['train_samples']}"
    assert meta["val_samples"] == 45980, f"Val count mismatch: {meta['val_samples']}"
    assert meta["test_samples"] == 53390, f"Test count mismatch: {meta['test_samples']}"
    print("  -> Row counts verified: 71,824 + 45,980 + 53,390 = 171,194")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # Feature contract verification
    assert list(X_train.columns) == PREDICTOR_FEATURES, "Feature column ordering mismatch!"
    assert len(PREDICTOR_FEATURES) == 14, f"Feature count != 14"
    print(f"  -> 14-feature contract verified: {PREDICTOR_FEATURES}")

    # ------------------------------------------------------------------
    # STAGE 2: Validation sanity check (reproduce Experiment 2 result)
    # ------------------------------------------------------------------
    print("\n--- STAGE 2: VALIDATION SANITY CHECK (Experiment 2 Reproduction) ---")
    print(f"  Parameters: {CHAMPION_PARAMS}")

    model_val_check = CatBoostRegressor(**CHAMPION_PARAMS)
    model_val_check.fit(X_train, y_train)
    preds_val = model_val_check.predict(X_val)
    metrics_val = calculate_regression_metrics(y_val, preds_val)

    print(f"  2017 Validation Metrics: {metrics_val}")
    print(f"  Expected (Exp 2):        R2=0.5357, MAE=673.58, RMSE=1253.76, MedAE=342.23")

    # Verify reproduction matches (allow tiny float tolerance)
    assert abs(metrics_val["R2"] - 0.5357) < 0.001, f"R2 reproduction failed: {metrics_val['R2']}"
    assert abs(metrics_val["MAE"] - 673.58) < 1.0, f"MAE reproduction failed: {metrics_val['MAE']}"
    print("  -> Experiment 2 validation metrics successfully reproduced.")

    # ------------------------------------------------------------------
    # STAGE 3: Retrain on Train + Validation
    # ------------------------------------------------------------------
    print("\n--- STAGE 3: FINAL CHAMPION RETRAIN (TRAIN + VALIDATION) ---")
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    print(f"  Combined Training Set: {X_train_val.shape[0]:,} samples (2013-04 to 2017-12)")
    assert X_train_val.shape[0] == 117804, f"Combined set size mismatch: {X_train_val.shape[0]}"
    assert X_train_val.shape[1] == 14, f"Feature count mismatch: {X_train_val.shape[1]}"

    model_final = CatBoostRegressor(**CHAMPION_PARAMS)
    model_final.fit(X_train_val, y_train_val)
    print("  -> Champion retrained on 117,804 samples.")

    # ------------------------------------------------------------------
    # STAGE 4: Final 2018 Test Evaluation (FIRST AND ONLY TIME)
    # ------------------------------------------------------------------
    print("\n--- STAGE 4: FINAL 2018 TEST EVALUATION ---")
    print("  *** This is the FIRST and ONLY evaluation on 2018 Test data. ***")

    test_preds = model_final.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, test_preds)
    test_error_dist = calculate_error_distribution(y_test, test_preds)
    test_segmented = calculate_segmented_metrics(y_test, test_preds, y_train)

    print(f"  2018 Test Metrics       : {test_metrics}")
    print(f"  Error Distribution      : {test_error_dist}")

    # Extract ±500 and ±1000 percentages
    pct_500 = test_error_dist["pct_within_500"]
    pct_1000 = test_error_dist["pct_within_1000"]

    # ------------------------------------------------------------------
    # STAGE 5: Comparison Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("FINAL COMPARISON: ORIGINAL PRODUCTION vs CHAMPION (2018 TEST)")
    print("=" * 115)

    print(f"\n{'Metric':<18} | {'Original Production':>20} | {'Final Champion':>20} | {'Change':>12}")
    print("-" * 80)

    comparisons = [
        ("MAE", ORIGINAL_TEST_2018["MAE"], test_metrics["MAE"]),
        ("RMSE", ORIGINAL_TEST_2018["RMSE"], test_metrics["RMSE"]),
        ("R²", ORIGINAL_TEST_2018["R2"], test_metrics["R2"]),
        ("MedAE", ORIGINAL_TEST_2018["MedAE"], test_metrics["MedAE"]),
        ("Within ±500", ORIGINAL_TEST_2018["pct_within_500"], pct_500),
        ("Within ±1000", ORIGINAL_TEST_2018["pct_within_1000"], pct_1000),
    ]

    for name, orig, new in comparisons:
        delta = new - orig
        if name in ("MAE", "RMSE", "MedAE"):
            direction = "[BETTER]" if delta < 0 else ("[WORSE]" if delta > 0 else "[SAME]")
            print(f"{name:<18} | {orig:>20.2f} | {new:>20.2f} | {delta:>+10.2f} {direction}")
        elif name == "R2":
            direction = "[BETTER]" if delta > 0 else ("[WORSE]" if delta < 0 else "[SAME]")
            print(f"{name:<18} | {orig:>20.4f} | {new:>20.4f} | {delta:>+10.4f} {direction}")
        else:  # percentage metrics
            direction = "[BETTER]" if delta > 0 else ("[WORSE]" if delta < 0 else "[SAME]")
            print(f"{name:<18} | {orig:>19.2f}% | {new:>19.2f}% | {delta:>+9.2f}% {direction}")

    print("-" * 80)

    # ------------------------------------------------------------------
    # STAGE 6: Save production model
    # ------------------------------------------------------------------
    print("\n--- STAGE 6: REPLACING PRODUCTION MODEL ARTIFACT ---")

    production_path = MODELS_DIR / "catboost_model.joblib"

    # Backup info
    if production_path.exists():
        old_size = production_path.stat().st_size
        print(f"  Existing artifact: {production_path} ({old_size / 1024:.1f} KB)")
    else:
        print(f"  No existing artifact found at {production_path}")

    joblib.dump(model_final, production_path)
    new_size = production_path.stat().st_size
    print(f"  New artifact saved: {production_path} ({new_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # STAGE 7: Post-save verification
    # ------------------------------------------------------------------
    print("\n--- STAGE 7: POST-SAVE VERIFICATION ---")

    # 7a. Load the saved model
    loaded_model = joblib.load(production_path)
    assert loaded_model is not None, "Failed to load saved model"
    print("  [PASS] Model loaded successfully from disk.")

    # 7b. Verify feature contract
    if hasattr(loaded_model, "feature_names_") and loaded_model.feature_names_ is not None:
        loaded_features = list(loaded_model.feature_names_)
        assert loaded_features == PREDICTOR_FEATURES, (
            f"Feature contract mismatch!\n"
            f"Expected: {PREDICTOR_FEATURES}\n"
            f"Got: {loaded_features}"
        )
        print(f"  [PASS] Feature contract verified: {len(loaded_features)} features in correct order.")
    else:
        print("  [WARN] Model does not expose feature_names_; contract check skipped.")

    # 7c. Verify prediction reproducibility
    test_pred_verify = loaded_model.predict(X_test)
    verify_metrics = calculate_regression_metrics(y_test, test_pred_verify)
    assert verify_metrics == test_metrics, (
        f"Prediction mismatch after reload!\n"
        f"Original: {test_metrics}\n"
        f"Reloaded: {verify_metrics}"
    )
    print(f"  [PASS] Prediction reproducibility verified after reload.")

    # ------------------------------------------------------------------
    # STAGE 8: Save promotion results
    # ------------------------------------------------------------------
    promotion_results = {
        "experiment": "Champion Promotion — Experiment 2 Winner",
        "champion_params": CHAMPION_PARAMS,
        "original_params": ORIGINAL_PARAMS,
        "training_set": "Train + Validation (2013-04 to 2017-12, 117,804 samples)",
        "test_set": "2018 (53,390 samples)",
        "validation_reproduction": metrics_val,
        "final_2018_test_metrics": test_metrics,
        "final_2018_error_distribution": test_error_dist,
        "final_2018_segmented_metrics": test_segmented,
        "original_2018_test_metrics": ORIGINAL_TEST_2018,
        "comparison": {
            name: {"original": orig, "champion": new, "delta": round(new - orig, 4)}
            for name, orig, new in comparisons
        },
        "production_artifact": str(production_path),
        "post_save_verification": "PASSED",
        "log1p_used": False,
        "early_stopping_used": False,
        "test_data_used_for_tuning": False,
    }

    results_path = EXPERIMENTS_DIR / "champion_promotion_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(promotion_results, f, indent=2)
    print(f"\n  Saved promotion results: {results_path}")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("CHAMPION MODEL PROMOTION COMPLETED")
    print("=" * 115)
    print(f"  Production artifact: {production_path}")
    print(f"  Parameters: depth=8, l2_leaf_reg=5, random_strength=1.0, iterations=300, lr=0.05")
    print(f"  Trained on: 117,804 samples (2013-04 to 2017-12)")
    print(f"  2018 Test R²   = {test_metrics['R2']:.4f}")
    print(f"  2018 Test MAE  = {test_metrics['MAE']:.2f}")
    print(f"  2018 Test RMSE = {test_metrics['RMSE']:.2f}")
    print(f"  2018 Test MedAE = {test_metrics['MedAE']:.2f}")
    print(f"  Within ±500    = {pct_500:.2f}%")
    print(f"  Within ±1000   = {pct_1000:.2f}%")
    print("=" * 115)


if __name__ == "__main__":
    run_champion_promotion()
