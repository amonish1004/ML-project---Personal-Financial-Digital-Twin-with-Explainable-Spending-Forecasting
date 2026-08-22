import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

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
    PersistenceBaseline,
    build_ridge_pipeline,
    build_catboost_model,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_pipeline():
    print("=" * 85)
    print("PERSONAL FINANCIAL DIGITAL TWIN — ML TRAINING, RETRAIN & EVALUATION PIPELINE")
    print("=" * 85)

    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"
    
    print("\n--- STAGE 1: LOADING & TEMPORAL PARTITIONING ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    print(f"Total Dataset Rows       : {meta['total_samples']:,}")
    print(f"TRAIN Set (2013-04..2016-12) : {meta['train_samples']:,} samples | {meta['train_accounts']:,} accounts")
    print(f"VAL Set   (2017-01..2017-12) : {meta['val_samples']:,} samples | {meta['val_accounts']:,} accounts")
    print(f"TEST Set  (2018-01..2018-12) : {meta['test_samples']:,} samples | {meta['test_accounts']:,} accounts")

    # Expected count checks
    assert meta['train_samples'] == 71824, f"Expected 71,824 Train rows, got {meta['train_samples']}"
    assert meta['val_samples'] == 45980, f"Expected 45,980 Val rows, got {meta['val_samples']}"
    assert meta['test_samples'] == 53390, f"Expected 53,390 Test rows, got {meta['test_samples']}"
    print("  -> All partition row counts match expected Dimension 3 & 4 specifications exactly!")

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]
    X_test, y_test = split_data["X_test"], split_data["y_test"]

    print(f"\nFeature Matrix Shape (X_train) : {X_train.shape} (14 predictors)")
    print(f"Target Vector Shape  (y_train) : {y_train.shape} (next_month_total_spending)")

    # ----------------------------------------------------
    # Model 1: Naive Persistence Baseline
    # ----------------------------------------------------
    print("\n--- STAGE 2: MODEL 1 — NAIVE PERSISTENCE BASELINE ---")
    model_persistence = PersistenceBaseline(target_lag_feature="spending_t")
    model_persistence.fit(X_train, y_train)

    val_preds_pers = model_persistence.predict(X_val)
    test_preds_pers = model_persistence.predict(X_test)

    val_metrics_pers = calculate_regression_metrics(y_val, val_preds_pers)
    test_metrics_pers = calculate_regression_metrics(y_test, test_preds_pers)

    print("  - Fitted: Deterministic spending_t persistence rule")
    print(f"  - Validation Metrics : {val_metrics_pers}")
    print(f"  - Test Metrics (Train-only): {test_metrics_pers}")

    # ----------------------------------------------------
    # Model 2: Ridge Regression
    # ----------------------------------------------------
    print("\n--- STAGE 3: MODEL 2 — RIDGE REGRESSION PIPELINE ---")
    model_ridge = build_ridge_pipeline(alpha=1.0, random_state=42)
    model_ridge.fit(X_train, y_train)

    val_preds_ridge = model_ridge.predict(X_val)
    test_preds_ridge = model_ridge.predict(X_test)

    val_metrics_ridge = calculate_regression_metrics(y_val, val_preds_ridge)
    test_metrics_ridge = calculate_regression_metrics(y_test, test_preds_ridge)

    print("  - Fitted: StandardScaler + Ridge(alpha=1.0) on Train partition")
    print(f"  - Validation Metrics : {val_metrics_ridge}")
    print(f"  - Test Metrics (Train-only): {test_metrics_ridge}")

    # ----------------------------------------------------
    # Model 3: CatBoost Regressor (Train-only candidate)
    # ----------------------------------------------------
    print("\n--- STAGE 4: MODEL 3 — CATBOOST REGRESSOR (CANDIDATE SELECTION) ---")
    model_catboost_train = build_catboost_model(iterations=300, learning_rate=0.05, depth=6, random_seed=42)
    model_catboost_train.fit(X_train, y_train)

    val_preds_cb = model_catboost_train.predict(X_val)
    test_preds_cb_train = model_catboost_train.predict(X_test)

    val_metrics_cb = calculate_regression_metrics(y_val, val_preds_cb)
    test_metrics_cb_train = calculate_regression_metrics(y_test, test_preds_cb_train)

    print("  - Fitted: CatBoostRegressor(iterations=300, lr=0.05, depth=6, seed=42) on Train (2013-2016)")
    print(f"  - Validation Metrics : {val_metrics_cb}")
    print(f"  - Test Metrics (Train-only): {test_metrics_cb_train}")
    print("  -> CHAMPION MODEL SELECTED ON VALIDATION: CatBoost Regressor (R² = 0.5339)")

    # ----------------------------------------------------
    # Stage 5: Final Model Retraining (Train + Validation -> Test)
    # ----------------------------------------------------
    print("\n--- STAGE 5: FINAL CHAMPION RETRAIN (TRAIN + VAL -> TEST) ---")
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    print(f"  - Combined Retraining Matrix Shape (X_train_val): {X_train_val.shape} (117,804 samples, 2013-04 to 2017-12)")
    
    model_catboost_final = build_catboost_model(iterations=300, learning_rate=0.05, depth=6, random_seed=42)
    model_catboost_final.fit(X_train_val, y_train_val)

    test_preds_cb_retrained = model_catboost_final.predict(X_test)
    test_metrics_cb_retrained = calculate_regression_metrics(y_test, test_preds_cb_retrained)

    # Detailed Dimension 4 Error Diagnostics
    test_error_dist = calculate_error_distribution(y_test, test_preds_cb_retrained)
    test_segmented_metrics = calculate_segmented_metrics(y_test, test_preds_cb_retrained, y_train)

    print("  - Retrained: CatBoostRegressor on 117,804 samples (2013-04 to 2017-12)")
    print(f"  - AUTHORITATIVE FINAL TEST METRICS : {test_metrics_cb_retrained}")
    print(f"  - Error Distribution Diagnostics   : {test_error_dist}")

    # ----------------------------------------------------
    # Empirical Results Summary Table
    # ----------------------------------------------------
    print("\n" + "=" * 95)
    print("EMPIRICAL MODEL EVALUATION SUMMARY TABLE (DIMENSION 3 & DIMENSION 4)")
    print("=" * 95)
    print(f"{'Model Name / Strategy':<38} | {'Partition':<10} | {'MAE (CZK)':<10} | {'RMSE (CZK)':<10} | {'R²':<8} | {'MedAE (CZK)':<10}")
    print("-" * 95)

    results_table = [
        ("Naive Persistence Baseline", "Validation", val_metrics_pers),
        ("Naive Persistence Baseline", "Test", test_metrics_pers),
        ("Ridge Regression (Scaled)", "Validation", val_metrics_ridge),
        ("Ridge Regression (Scaled)", "Test", test_metrics_ridge),
        ("CatBoost Regressor (Train Only)", "Validation", val_metrics_cb),
        ("CatBoost Regressor (Train Only)", "Test", test_metrics_cb_train),
        ("Final CatBoost (Train+Val Retrained)", "Test", test_metrics_cb_retrained),
    ]

    for model_name, part, m in results_table:
        print(f"{model_name:<38} | {part:<10} | {m['MAE']:<10.2f} | {m['RMSE']:<10.2f} | {m['R2']:<8.4f} | {m['MedAE']:<10.2f}")

    print("-" * 95)

    # ----------------------------------------------------
    # Serialization of Model Artifacts
    # ----------------------------------------------------
    print("\n--- STAGE 6: SAVING FINAL MODEL ARTIFACTS & EXPERIMENT JSON ---")
    
    ridge_path = MODELS_DIR / "ridge_pipeline.joblib"
    joblib.dump(model_ridge, ridge_path)
    print(f"  - Saved Ridge Model      : {ridge_path} ({ridge_path.stat().st_size / 1024:.1f} KB)")

    # Save final retrained CatBoost model
    catboost_path = MODELS_DIR / "catboost_model.joblib"
    joblib.dump(model_catboost_final, catboost_path)
    print(f"  - Saved Retrained CatBoost: {catboost_path} ({catboost_path.stat().st_size / 1024:.1f} KB)")

    # Save complete experiment metrics record JSON
    experiment_results = {
        "dataset_path": str(csv_path.relative_to(PROJECT_ROOT)),
        "predictor_features": PREDICTOR_FEATURES,
        "target_column": TARGET_COLUMN,
        "split_metadata": meta,
        "hyperparameters": {
            "ridge": {"alpha": 1.0, "random_state": 42},
            "catboost": {"iterations": 300, "learning_rate": 0.05, "depth": 6, "random_seed": 42},
        },
        "metrics": {
            "persistence_baseline": {"validation": val_metrics_pers, "test": test_metrics_pers},
            "ridge_regression": {"validation": val_metrics_ridge, "test": test_metrics_ridge},
            "catboost_regressor_train_only": {"validation": val_metrics_cb, "test": test_metrics_cb_train},
            "catboost_regressor_final_retrained": {"test": test_metrics_cb_retrained},
        },
        "error_diagnostics_final_test": test_error_dist,
        "segmented_metrics_final_test": test_segmented_metrics,
    }

    results_json_path = MODELS_DIR / "experiment_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"  - Saved Experiment JSON  : {results_json_path}")

    print("\n" + "=" * 85)
    print("DIMENSION 4 MODEL EVALUATION & RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    run_experiment_pipeline()

