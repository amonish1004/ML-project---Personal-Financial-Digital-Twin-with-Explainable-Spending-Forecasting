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


def run_boosting_params_experiment():
    print("=" * 115)
    print("PERSONAL FINANCIAL DIGITAL TWIN — ISOLATED CATBOOST BOOSTING SCHEDULE EXPERIMENT")
    print("=" * 115)

    csv_path = DATA_PROCESSED_DIR / "supervised_spending_dataset.csv"
    
    print("\n--- STAGE 1: LOADING & TEMPORAL PARTITIONING ---")
    split_data = load_and_split_dataset(csv_path)
    meta = split_data["metadata"]

    print(f"Total Dataset Rows           : {meta['total_samples']:,}")
    print(f"TRAIN Set (2013-04..2016-12)     : {meta['train_samples']:,} samples")
    print(f"VAL Set   (2017-01..2017-12)     : {meta['val_samples']:,} samples")
    print(f"TEST Set  (2018-01..2018-12)     : {meta['test_samples']:,} samples (HELD OUT - UNTOUCHED)")

    X_train, y_train = split_data["X_train"], split_data["y_train"]
    X_val, y_val = split_data["X_val"], split_data["y_val"]

    assert list(X_train.columns) == PREDICTOR_FEATURES, "Feature column ordering mismatch!"
    print(f"\nFeatures verified ({len(PREDICTOR_FEATURES)} predictors): {PREDICTOR_FEATURES}")

    # Reference Experiment 2 baseline winner: depth=8, l2=5, rs=1.0, iter=300, lr=0.05
    lr_grid = [0.02, 0.03, 0.05, 0.08]
    iter_grid = [300, 500, 800]

    fixed_params = {
        "depth": 8,
        "l2_leaf_reg": 5,
        "random_strength": 1.0,
        "random_seed": 42,
        "loss_function": "RMSE",
        "verbose": 0,
    }

    results = []

    print("\n--- STAGE 2: RUNNING 12-CONFIGURATION BOOSTING SCHEDULE GRID SEARCH ---")
    config_idx = 0
    for lr in lr_grid:
        for it in iter_grid:
            config_idx += 1
            is_baseline = (it == 300 and lr == 0.05)
            tag = " (EXP 2 BASELINE WINNER)" if is_baseline else ""

            model = CatBoostRegressor(
                iterations=it,
                learning_rate=lr,
                **fixed_params
            )
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            metrics = calculate_regression_metrics(y_val, val_preds)

            res_entry = {
                "id": config_idx,
                "iterations": it,
                "learning_rate": lr,
                "depth": fixed_params["depth"],
                "l2_leaf_reg": fixed_params["l2_leaf_reg"],
                "random_strength": fixed_params["random_strength"],
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "R2": float(metrics["R2"]),
                "MedAE": float(metrics["MedAE"]),
                "is_baseline": bool(is_baseline),
            }
            results.append(res_entry)
            print(f"Config {config_idx:2d}/12: iter={it:3d}, lr={lr:4.2f} -> R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}{tag}")

    df_res = pd.DataFrame(results)

    # Baseline row
    baseline_row = df_res[df_res["is_baseline"]].iloc[0]
    base_r2 = baseline_row["R2"]
    base_mae = baseline_row["MAE"]
    base_rmse = baseline_row["RMSE"]
    base_medae = baseline_row["MedAE"]

    # Table 1: Sorted by R² descending
    df_sorted_r2 = df_res.sort_values(by="R2", ascending=False).reset_index(drop=True)

    # Table 2: Sorted by MAE ascending
    df_sorted_mae = df_res.sort_values(by="MAE", ascending=True).reset_index(drop=True)

    # Table 3: Sorted by RMSE ascending
    df_sorted_rmse = df_res.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    print("\n" + "=" * 115)
    print("COMPLETE RESULTS TABLE — SORTED BY VALIDATION R² (DESCENDING)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Iter':<6} | {'LR':<6} | {'Depth':<5} | {'L2 Reg':<6} | {'MAE':<9} | {'RMSE':<9} | {'R2':<7} | {'MedAE':<9} | {'Notes':<24}")
    print("-" * 115)
    for idx, row in df_sorted_r2.iterrows():
        notes = "EXP 2 BASELINE" if row["is_baseline"] else ""
        print(f"{idx+1:<5} | {int(row['iterations']):<6} | {row['learning_rate']:<6.2f} | {int(row['depth']):<5} | {int(row['l2_leaf_reg']):<6} | {row['MAE']:<9.2f} | {row['RMSE']:<9.2f} | {row['R2']:<7.4f} | {row['MedAE']:<9.2f} | {notes:<24}")

    # Best configurations
    best_r2_row = df_sorted_r2.iloc[0]
    best_mae_row = df_sorted_mae.iloc[0]
    best_rmse_row = df_sorted_rmse.iloc[0]

    r2_abs_imp = best_r2_row["R2"] - base_r2
    r2_pct_imp = (r2_abs_imp / base_r2) * 100.0 if base_r2 != 0 else 0.0

    mae_abs_imp = base_mae - best_mae_row["MAE"]  # Positive means MAE decreased (improved)
    mae_pct_imp = (mae_abs_imp / base_mae) * 100.0 if base_mae != 0 else 0.0

    rmse_abs_imp = base_rmse - best_rmse_row["RMSE"]
    rmse_pct_imp = (rmse_abs_imp / base_rmse) * 100.0 if base_rmse != 0 else 0.0

    same_winner = (best_r2_row["id"] == best_mae_row["id"] == best_rmse_row["id"])

    # Qualitative Assessment
    if r2_abs_imp > 0.01:
        assessment_r2 = "Meaningful Improvement"
    elif r2_abs_imp > 0.001:
        assessment_r2 = "Small / Marginal Improvement"
    elif r2_abs_imp > 0:
        assessment_r2 = "Negligible Improvement"
    else:
        assessment_r2 = "Worse / No Improvement"

    print("\n" + "=" * 115)
    print("EXPERIMENT SUMMARY & METRIC LEADERBOARD")
    print("=" * 115)
    print(f"1. BASELINE REFERENCE METRICS (Exp 2 Winner: iter=300, lr=0.05, depth=8, l2=5):")
    print(f"   - R²: {base_r2:.4f} | MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f} | MedAE: {base_medae:.2f}")

    print(f"\n2. TOP 5 CONFIGURATIONS BY VALIDATION R²:")
    for i in range(min(5, len(df_sorted_r2))):
        r = df_sorted_r2.iloc[i]
        b_mark = " (BASELINE)" if r["is_baseline"] else ""
        print(f"   #{i+1}: iter={int(r['iterations']):3d}, lr={r['learning_rate']:.2f} -> R²={r['R2']:.4f}, MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}{b_mark}")

    print(f"\n3. TOP 5 CONFIGURATIONS BY VALIDATION MAE:")
    for i in range(min(5, len(df_sorted_mae))):
        r = df_sorted_mae.iloc[i]
        b_mark = " (BASELINE)" if r["is_baseline"] else ""
        print(f"   #{i+1}: iter={int(r['iterations']):3d}, lr={r['learning_rate']:.2f} -> MAE={r['MAE']:.2f}, R²={r['R2']:.4f}, RMSE={r['RMSE']:.2f}{b_mark}")

    print(f"\n4. BEST CONFIGURATION BY VALIDATION R²:")
    print(f"   - Parameters: iterations={int(best_r2_row['iterations'])}, learning_rate={best_r2_row['learning_rate']:.2f} (depth=8, l2=5, rs=1.0)")
    print(f"   - R²: {best_r2_row['R2']:.4f} | MAE: {best_r2_row['MAE']:.2f} | RMSE: {best_r2_row['RMSE']:.2f} | MedAE: {best_r2_row['MedAE']:.2f}")
    print(f"   - R² Improvement over Baseline: {r2_abs_imp:+.4f} ({base_r2:.4f} -> {best_r2_row['R2']:.4f}) [{r2_pct_imp:+.2f}% relative]")

    print(f"\n5. BEST CONFIGURATION BY VALIDATION MAE:")
    print(f"   - Parameters: iterations={int(best_mae_row['iterations'])}, learning_rate={best_mae_row['learning_rate']:.2f}")
    print(f"   - MAE: {best_mae_row['MAE']:.2f} | R²: {best_mae_row['R2']:.4f} | RMSE: {best_mae_row['RMSE']:.2f} | MedAE: {best_mae_row['MedAE']:.2f}")
    print(f"   - MAE Improvement over Baseline: {mae_abs_imp:+.2f} ({base_mae:.2f} -> {best_mae_row['MAE']:.2f}) [{mae_pct_imp:+.2f}% relative]")

    print(f"\n6. BEST CONFIGURATION BY VALIDATION RMSE:")
    print(f"   - Parameters: iterations={int(best_rmse_row['iterations'])}, learning_rate={best_rmse_row['learning_rate']:.2f}")
    print(f"   - RMSE: {best_rmse_row['RMSE']:.2f} | R²: {best_rmse_row['R2']:.4f} | MAE: {best_rmse_row['MAE']:.2f}")
    print(f"   - RMSE Improvement over Baseline: {rmse_abs_imp:+.2f} ({base_rmse:.2f} -> {best_rmse_row['RMSE']:.2f}) [{rmse_pct_imp:+.2f}% relative]")

    print(f"\n7. MULTI-METRIC CONVERGENCE:")
    print(f"   - Does the same configuration win R², MAE, and RMSE? {'YES' if same_winner else 'NO'}")

    print(f"\n8. QUALITATIVE ASSESSMENT OF IMPROVEMENT:")
    print(f"   - R² Delta: {r2_abs_imp:+.4f} -> Classification: {assessment_r2}")

    exp_summary_path = EXPERIMENTS_DIR / "boosting_params_results.json"
    with open(exp_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "configurations_tested": len(results),
            "baseline_exp2": baseline_row.to_dict(),
            "best_r2": best_r2_row.to_dict(),
            "best_mae": best_mae_row.to_dict(),
            "best_rmse": best_rmse_row.to_dict(),
            "qualitative_assessment": assessment_r2,
            "all_results": results
        }, f, indent=2)
    print(f"\n  - Saved Results JSON: {exp_summary_path}")

    print("\n" + "=" * 115)
    print("BOOSTING SCHEDULE EXPERIMENT COMPLETED — PRODUCTION MODEL WAS NOT TOUCHED.")
    print("=" * 115)


if __name__ == "__main__":
    run_boosting_params_experiment()
