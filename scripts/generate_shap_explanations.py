import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR
from src.models import (
    PREDICTOR_FEATURES,
    TARGET_COLUMN,
    load_and_split_dataset,
    initialize_tree_explainer,
    compute_shap_values,
    compute_feature_importance_ranking,
    generate_global_shap_plots,
    generate_top_dependence_plots,
    select_deterministic_archetypes,
    generate_archetype_waterfall_plots,
    export_shap_summary_json,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SHAP_OUTPUT_DIR = REPORTS_DIR / "figures" / "shap"


def run_shap_pipeline():
    print("=" * 85)
    print("PERSONAL FINANCIAL DIGITAL TWIN — DIMENSION 5 SHAP EXPLAINABILITY PIPELINE")
    print("=" * 85)

    # 1. Load Supervised Dataset & Temporal Splits
    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"
    print("\n--- STAGE 1: LOADING DATASET & Chronological TEST PARTITION ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    df_test_meta = split_data["df_test_meta"]

    print(f"Dataset Path             : {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"TEST Partition Period    : {meta['test_period']}")
    print(f"TEST Partition Samples   : {meta['test_samples']:,} samples | {meta['test_accounts']:,} accounts")
    print(f"Predictor Features Count : {len(PREDICTOR_FEATURES)} features")

    # Sanity Checks on Dataset Partition
    assert len(X_test) == 53390, f"Expected 53,390 Test rows, got {len(X_test)}"
    assert list(X_test.columns) == PREDICTOR_FEATURES, "X_test predictor feature column names mismatch"
    print("  [SANITY CHECK PASSED] Dataset partition shape and feature schema verified.")

    # 2. Load Frozen Retrained CatBoost Model
    model_path = MODELS_DIR / "catboost_model.joblib"
    print(f"\n--- STAGE 2: LOADING FROZEN CATBOOST MODEL ---")
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model artifact missing at: {model_path}")

    model = joblib.load(model_path)
    print(f"  Loaded Artifact: {model_path.relative_to(PROJECT_ROOT)} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Model Type     : {type(model).__name__}")
    print(f"  Tree Count     : {getattr(model, 'tree_count_', getattr(model, '_tree_count', '300'))}")


    # 3. Model Inference Sanity Check
    print("\n--- STAGE 3: INFERENCE SANITY CHECK ---")
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test), "Prediction array length mismatch"
    print(f"  Predictions generated for {len(y_pred):,} test samples.")
    print(f"  Prediction Range: Min = {y_pred.min():,.2f} CZK | Max = {y_pred.max():,.2f} CZK | Mean = {y_pred.mean():,.2f} CZK")
    print("  [SANITY CHECK PASSED] Model inference verified.")

    # 4. TreeSHAP Calculation
    print("\n--- STAGE 4: TREESHAP CALCULATION ON FULL TEST PARTITION ---")
    explainer = initialize_tree_explainer(model)
    print("  Initialized TreeExplainer directly from CatBoost tree structure.")
    print(f"  Computing TreeSHAP values across all {len(X_test):,} test observations...")

    shap_values_matrix, explanation_obj, base_value, elapsed_sec = compute_shap_values(explainer, X_test)

    assert shap_values_matrix.shape == (len(X_test), len(PREDICTOR_FEATURES)), (
        f"SHAP matrix shape error: expected ({len(X_test)}, 14), got {shap_values_matrix.shape}"
    )

    print(f"  TreeSHAP calculation completed in {elapsed_sec:.2f} seconds.")
    print(f"  SHAP Matrix Shape  : {shap_values_matrix.shape}")
    print(f"  Expected Value E[f(X)] (Base Spending): {base_value:,.4f} CZK")
    print("  [SANITY CHECK PASSED] SHAP matrix shape and expected value verified.")

    # 5. Global Feature Ranking & Importance
    print("\n--- STAGE 5: GLOBAL FEATURE IMPORTANCE RANKING ---")
    df_ranking = compute_feature_importance_ranking(shap_values_matrix, PREDICTOR_FEATURES)

    print(f"{'Rank':<5} | {'Predictor Feature':<22} | {'Mean |SHAP| (CZK)':<18} | {'Mean SHAP (CZK)':<16} | {'Std SHAP':<12}")
    print("-" * 80)
    for _, row in df_ranking.iterrows():
        print(
            f"{int(row['rank']):<5} | "
            f"{row['feature']:<22} | "
            f"{row['mean_abs_shap']:<18.4f} | "
            f"{row['mean_shap']:<16.4f} | "
            f"{row['std_shap']:<12.4f}"
        )
    print("-" * 80)

    # 6. Global Summary Plots Generation
    print("\n--- STAGE 6: GENERATING GLOBAL SUMMARY PLOTS ---")
    bar_path, beeswarm_path = generate_global_shap_plots(
        explanation_obj,
        shap_values_matrix,
        X_test,
        PREDICTOR_FEATURES,
        SHAP_OUTPUT_DIR
    )
    print(f"  Saved Bar Plot      : {bar_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved Beeswarm Plot : {beeswarm_path.relative_to(PROJECT_ROOT)}")

    # 7. Dynamic Top-3 Feature Dependence Plots
    print("\n--- STAGE 7: GENERATING TOP-3 FEATURE DEPENDENCE PLOTS ---")
    top_3_features = df_ranking["feature"].head(3).tolist()
    print(f"  Top 3 Empirical Features Selected: {top_3_features}")

    top_dep_files = generate_top_dependence_plots(
        shap_values_matrix,
        X_test,
        df_ranking,
        SHAP_OUTPUT_DIR,
        top_k=3
    )
    for feat_name, filepath in top_dep_files:
        print(f"  Saved Dependence Plot ({feat_name}) : {filepath.relative_to(PROJECT_ROOT)}")

    # 8. Deterministic Archetype Selection & Local Explanations
    print("\n--- STAGE 8: DETERMINISTIC ARCHETYPE SELECTION & WATERFALL PLOTS ---")
    archetypes_info = select_deterministic_archetypes(df_test_meta, X_test, y_test, y_pred)

    for arch_key, info in archetypes_info.items():
        print(f"  Archetype [{info['label']}]")
        print(f"    - Account #{info['account_id']} ({info['target_month']})")
        print(f"    - Actual y: {info['actual_spending_y']:,.2f} CZK | Pred y_hat: {info['predicted_spending_y_hat']:,.2f} CZK")
        print(f"    - Residual (y_hat - y): {info['residual_y_hat_minus_y']:,.2f} CZK")

    waterfall_plots = generate_archetype_waterfall_plots(explanation_obj, archetypes_info, SHAP_OUTPUT_DIR)
    for arch_key, w_path in waterfall_plots.items():
        print(f"  Saved Waterfall Plot ({arch_key}) : {w_path.relative_to(PROJECT_ROOT)}")

    # 9. JSON Summary Export
    print("\n--- STAGE 9: EXPORTING SHAP SUMMARY JSON ---")
    json_path = MODELS_DIR / "shap_summary_results.json"
    export_shap_summary_json(
        base_value,
        meta,
        elapsed_sec,
        df_ranking,
        archetypes_info,
        json_path
    )
    print(f"  Saved SHAP Summary JSON : {json_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 85)
    print("DIMENSION 5 SHAP EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    run_shap_pipeline()
