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


def run_hyperparameter_experiment():
    print("=" * 115)
    print("PERSONAL FINANCIAL DIGITAL TWIN — ISOLATED CATBOOST HYPERPARAMETER GRID EXPERIMENT")
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

    # Grid search space (24 combinations)
    depth_grid = [4, 6, 8]
    l2_grid = [1, 3, 5, 10]
    rs_grid = [0.1, 1.0]

    results = []

    print("\n--- STAGE 2: RUNNING 24-CONFIGURATION GRID SEARCH ---")
    config_idx = 0
    for d in depth_grid:
        for l2 in l2_grid:
            for rs in rs_grid:
                config_idx += 1
                is_baseline = (d == 6 and l2 == 3 and rs == 1.0)
                tag = " (BASELINE)" if is_baseline else ""
                
                model = CatBoostRegressor(
                    iterations=300,
                    learning_rate=0.05,
                    depth=d,
                    l2_leaf_reg=l2,
                    random_strength=rs,
                    random_seed=42,
                    loss_function="RMSE",
                    verbose=0,
                )
                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                metrics = calculate_regression_metrics(y_val, val_preds)

                res_entry = {
                    "id": config_idx,
                    "depth": d,
                    "l2_leaf_reg": l2,
                    "random_strength": rs,
                    "iterations": 300,
                    "learning_rate": 0.05,
                    "MAE": float(metrics["MAE"]),
                    "RMSE": float(metrics["RMSE"]),
                    "R2": float(metrics["R2"]),
                    "MedAE": float(metrics["MedAE"]),
                    "is_baseline": bool(is_baseline),
                }
                results.append(res_entry)
                print(f"Config {config_idx:2d}/24: depth={d}, l2={l2:2d}, rs={rs:3.1f} -> R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}{tag}")

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

    print("\n" + "=" * 115)
    print("COMPLETE RESULTS TABLE — SORTED BY VALIDATION R² (DESCENDING)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Depth':<5} | {'L2 Reg':<6} | {'RandStr':<7} | {'Iter':<5} | {'LR':<5} | {'MAE':<9} | {'RMSE':<9} | {'R2':<7} | {'MedAE':<9} | {'Notes':<12}")
    print("-" * 115)
    for idx, row in df_sorted_r2.iterrows():
        notes = "BASELINE" if row["is_baseline"] else ""
        print(f"{idx+1:<5} | {int(row['depth']):<5} | {int(row['l2_leaf_reg']):<6} | {row['random_strength']:<7.1f} | {int(row['iterations']):<5} | {row['learning_rate']:<5.2f} | {row['MAE']:<9.2f} | {row['RMSE']:<9.2f} | {row['R2']:<7.4f} | {row['MedAE']:<9.2f} | {notes:<12}")

    print("\n" + "=" * 115)
    print("COMPLETE RESULTS TABLE — SORTED BY VALIDATION MAE (ASCENDING)")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Depth':<5} | {'L2 Reg':<6} | {'RandStr':<7} | {'Iter':<5} | {'LR':<5} | {'MAE':<9} | {'RMSE':<9} | {'R2':<7} | {'MedAE':<9} | {'Notes':<12}")
    print("-" * 115)
    for idx, row in df_sorted_mae.iterrows():
        notes = "BASELINE" if row["is_baseline"] else ""
        print(f"{idx+1:<5} | {int(row['depth']):<5} | {int(row['l2_leaf_reg']):<6} | {row['random_strength']:<7.1f} | {int(row['iterations']):<5} | {row['learning_rate']:<5.2f} | {row['MAE']:<9.2f} | {row['RMSE']:<9.2f} | {row['R2']:<7.4f} | {row['MedAE']:<9.2f} | {notes:<12}")

    # Best configurations
    best_r2_row = df_sorted_r2.iloc[0]
    best_mae_row = df_sorted_mae.iloc[0]
    best_rmse_row = df_res.sort_values(by="RMSE", ascending=True).iloc[0]

    r2_imp = best_r2_row["R2"] - base_r2
    mae_imp = base_mae - best_mae_row["MAE"]

    same_winner = (best_r2_row["id"] == best_mae_row["id"] == best_rmse_row["id"])

    print("\n" + "=" * 115)
    print("EXPERIMENT SUMMARY & METRIC LEADERBOARD")
    print("=" * 115)
    print(f"1. BASELINE VALIDATION METRICS:")
    print(f"   - R²: {base_r2:.4f} | MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f} | MedAE: {base_medae:.2f}")

    print(f"\n2. BEST CONFIGURATION BY VALIDATION R²:")
    print(f"   - Parameters: depth={int(best_r2_row['depth'])}, l2_leaf_reg={int(best_r2_row['l2_leaf_reg'])}, random_strength={best_r2_row['random_strength']}")
    print(f"   - R²: {best_r2_row['R2']:.4f} | MAE: {best_r2_row['MAE']:.2f} | RMSE: {best_r2_row['RMSE']:.2f} | MedAE: {best_r2_row['MedAE']:.2f}")
    print(f"   - R² Improvement over Baseline: {r2_imp:+.4f} ({base_r2:.4f} -> {best_r2_row['R2']:.4f})")

    print(f"\n3. BEST CONFIGURATION BY VALIDATION MAE:")
    print(f"   - Parameters: depth={int(best_mae_row['depth'])}, l2_leaf_reg={int(best_mae_row['l2_leaf_reg'])}, random_strength={best_mae_row['random_strength']}")
    print(f"   - R²: {best_mae_row['R2']:.4f} | MAE: {best_mae_row['MAE']:.2f} | RMSE: {best_mae_row['RMSE']:.2f} | MedAE: {best_mae_row['MedAE']:.2f}")
    print(f"   - MAE Improvement over Baseline: {mae_imp:+.2f} ({base_mae:.2f} -> {best_mae_row['MAE']:.2f})")

    print(f"\n4. BEST CONFIGURATION BY VALIDATION RMSE:")
    print(f"   - Parameters: depth={int(best_rmse_row['depth'])}, l2_leaf_reg={int(best_rmse_row['l2_leaf_reg'])}, random_strength={best_rmse_row['random_strength']}")
    print(f"   - RMSE: {best_rmse_row['RMSE']:.2f} | R²: {best_rmse_row['R2']:.4f} | MAE: {best_rmse_row['MAE']:.2f}")

    print(f"\n5. MULTI-METRIC CONVERGENCE:")
    print(f"   - Does the same configuration win R², MAE, and RMSE? {'YES' if same_winner else 'NO'}")

    print(f"\n6. TOP 5 CONFIGURATIONS BY VALIDATION R²:")
    for i in range(min(5, len(df_sorted_r2))):
        r = df_sorted_r2.iloc[i]
        print(f"   #{i+1}: depth={int(r['depth'])}, l2={int(r['l2_leaf_reg'])}, rs={r['random_strength']} -> R²={r['R2']:.4f}, MAE={r['MAE']:.2f}, RMSE={r['RMSE']:.2f}")

    exp_summary_path = EXPERIMENTS_DIR / "hyperparameter_grid_results.json"
    with open(exp_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "configurations_tested": len(results),
            "baseline": baseline_row.to_dict(),
            "best_r2": best_r2_row.to_dict(),
            "best_mae": best_mae_row.to_dict(),
            "best_rmse": best_rmse_row.to_dict(),
            "all_results": results
        }, f, indent=2)
    print(f"\n  - Saved Grid Results JSON: {exp_summary_path}")

    print("\n" + "=" * 115)
    print("HYPERPARAMETER GRID EXPERIMENT COMPLETED — PRODUCTION MODEL WAS NOT TOUCHED.")
    print("=" * 115)


if __name__ == "__main__":
    run_hyperparameter_experiment()
