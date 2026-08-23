import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor


def initialize_tree_explainer(model: CatBoostRegressor) -> shap.TreeExplainer:
    """
    Initializes a TreeExplainer directly from a trained CatBoostRegressor model.
    CatBoost models provide exact tree structures for analytical TreeSHAP calculation.
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame
) -> Tuple[np.ndarray, Any, float, float]:
    """
    Computes SHAP values on the input feature matrix X.
    
    Returns
    -------
    shap_values_matrix : np.ndarray of shape (N, num_features)
    explanation_obj : shap.Explanation object (useful for modern waterfall & beeswarm plots)
    base_value : float (expected target value E[f(X)])
    elapsed_sec : float (computation time in seconds)
    """
    start_time = time.time()
    explanation_obj = explainer(X)
    elapsed_sec = round(time.time() - start_time, 4)

    if isinstance(explanation_obj.values, np.ndarray):
        shap_values_matrix = explanation_obj.values
    else:
        shap_values_matrix = np.array(explanation_obj.values)

    # Base value / Expected value E[f(X)]
    if hasattr(explanation_obj, "base_values"):
        bv = explanation_obj.base_values
        base_value = float(bv[0]) if isinstance(bv, (np.ndarray, list)) else float(bv)
    else:
        base_value = float(explainer.expected_value)

    return shap_values_matrix, explanation_obj, base_value, elapsed_sec


def compute_feature_importance_ranking(
    shap_values_matrix: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes global mean absolute SHAP values across all samples for each feature.
    Ranks features from 1 (most important) to N (least important).
    """
    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    mean_shap = shap_values_matrix.mean(axis=0)
    std_shap = shap_values_matrix.std(axis=0)

    df_rank = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "std_shap": std_shap,
    })

    df_rank.sort_values(by="mean_abs_shap", ascending=False, inplace=True)
    df_rank.reset_index(drop=True, inplace=True)
    df_rank["rank"] = df_rank.index + 1
    return df_rank


def generate_global_shap_plots(
    explanation_obj: Any,
    shap_values_matrix: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Generates and saves global SHAP summary plots:
    1. Global mean |SHAP| feature importance bar plot.
    2. Beeswarm distribution plot showing directional feature value impacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bar_path = output_dir / "shap_summary_bar.png"
    beeswarm_path = output_dir / "shap_summary_beeswarm.png"

    # 1. Global Mean |SHAP| Bar Plot
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(
        shap_values_matrix,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("Global Feature Importance — CatBoost Spending Model (Mean |SHAP|)", fontsize=12, pad=15)
    plt.xlabel("Mean |SHAP Value| (Impact on Next-Month Spending Forecast in CZK)", fontsize=10)
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight", dpi=300)
    plt.close("all")

    # 2. Beeswarm Distribution Plot
    plt.figure(figsize=(11, 7), dpi=300)
    shap.summary_plot(
        shap_values_matrix,
        X,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    plt.title("SHAP Beeswarm Distribution — Feature Value vs Spending Attribution", fontsize=12, pad=15)
    plt.xlabel("SHAP Value (Attribution to Next-Month Spending Forecast in CZK)", fontsize=10)
    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    plt.close("all")

    return bar_path, beeswarm_path


def generate_top_dependence_plots(
    shap_values_matrix: np.ndarray,
    X: pd.DataFrame,
    feature_ranking_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 3
) -> List[Tuple[str, Path]]:
    """
    Generates SHAP dependence plots for top-K empirical features.
    Features are selected dynamically based on their actual mean |SHAP| ranking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    top_features = feature_ranking_df["feature"].head(top_k).tolist()

    for idx, feature_name in enumerate(top_features, 1):
        plot_path = output_dir / f"shap_dependence_rank{idx}_{feature_name}.png"
        plt.figure(figsize=(9, 6), dpi=300)

        feature_idx = list(X.columns).index(feature_name)
        shap.dependence_plot(
            feature_idx,
            shap_values_matrix,
            X,
            feature_names=list(X.columns),
            show=False
        )

        plt.title(f"SHAP Dependence Plot — Rank #{idx}: {feature_name}", fontsize=12, pad=15)
        plt.ylabel(f"SHAP Value for {feature_name} (CZK Attribution)", fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close("all")

        generated_files.append((feature_name, plot_path))

    return generated_files


def select_deterministic_archetypes(
    df_test_meta: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Selects four deterministic, reproducible financial account archetypes from the TEST partition.
    
    Residual Convention: residual = y_pred - y_true
    - under-prediction: y_pred < y_true -> residual < 0 (negative residual)
    - over-prediction: y_pred > y_true -> residual > 0 (positive residual)
    
    Archetype Rules:
    1. Typical / Median Spender: Instance whose actual spending (y_test) is closest to median(y_test)
       among instances with near-zero error (|residual| <= 100 CZK).
    2. High Spender: Instance with maximum actual spending (y_test).
    3. Largest Under-Prediction: Instance with minimum residual (y_pred - y_true),
       where forecast was farthest below actual spending.
    4. Largest Over-Prediction: Instance with maximum residual (y_pred - y_true),
       where forecast was farthest above actual spending.
    """
    residuals = y_pred - y_test
    med_y = np.median(y_test)

    # 1. Typical / Median Spender
    abs_res = np.abs(residuals)
    small_error_indices = np.where(abs_res <= 100.0)[0]
    if len(small_error_indices) == 0:
        small_error_indices = np.argsort(abs_res)[:100]

    diff_to_med = np.abs(y_test[small_error_indices] - med_y)
    typical_idx = int(small_error_indices[np.argmin(diff_to_med)])

    # 2. High Spender
    high_spender_idx = int(np.argmax(y_test))

    # 3. Largest Under-Prediction (min residual: y_pred - y_true < 0)
    largest_under_idx = int(np.argmin(residuals))

    # 4. Largest Over-Prediction (max residual: y_pred - y_true > 0)
    largest_over_idx = int(np.argmax(residuals))

    archetype_keys = {
        "typical_median_spender": ("Typical / Median Spender", typical_idx),
        "high_spender": ("High Spender", high_spender_idx),
        "largest_under_prediction": ("Largest Under-Prediction (y_pred < y_true)", largest_under_idx),
        "largest_over_prediction": ("Largest Over-Prediction (y_pred > y_true)", largest_over_idx),
    }

    archetypes_info = {}
    for key, (label, idx) in archetype_keys.items():
        meta_row = df_test_meta.iloc[idx]
        account_id = int(meta_row["account_id"])
        ref_month = str(meta_row["reference_month"])
        target_month = str(meta_row["target_month"])
        actual_val = float(y_test[idx])
        pred_val = float(y_pred[idx])
        res_val = float(residuals[idx])

        archetypes_info[key] = {
            "label": label,
            "sample_index": idx,
            "account_id": account_id,
            "reference_month": ref_month,
            "target_month": target_month,
            "actual_spending_y": round(actual_val, 2),
            "predicted_spending_y_hat": round(pred_val, 2),
            "residual_y_hat_minus_y": round(res_val, 2),
            "feature_values": {col: float(X_test.iloc[idx][col]) for col in X_test.columns}
        }

    return archetypes_info


def generate_archetype_waterfall_plots(
    explanation_obj: Any,
    archetypes_info: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Generates local SHAP waterfall plots for each selected archetype instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    for key, info in archetypes_info.items():
        idx = info["sample_index"]
        plot_path = output_dir / f"shap_waterfall_{key}.png"

        plt.figure(figsize=(10, 6), dpi=300)
        shap.plots.waterfall(explanation_obj[idx], show=False)

        title_str = (
            f"SHAP Local Explanation — {info['label']}\n"
            f"Account #{info['account_id']} ({info['target_month']}) | "
            f"Actual: {info['actual_spending_y']:,.2f} CZK | Pred: {info['predicted_spending_y_hat']:,.2f} CZK"
        )
        plt.title(title_str, fontsize=11, pad=15)
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close("all")

        generated_plots[key] = plot_path

    return generated_plots


def export_shap_summary_json(
    base_value: float,
    dataset_metadata: Dict[str, Any],
    computation_time_sec: float,
    feature_ranking_df: pd.DataFrame,
    archetypes_info: Dict[str, Dict[str, Any]],
    output_path: Path
):
    """
    Serializes all numerical SHAP results, feature ranks, base value E[f(X)], and archetype metrics to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "dataset_evaluated": "TEST partition (2018-01 through 2018-12)",
        "num_test_samples": int(dataset_metadata["test_samples"]),
        "num_predictor_features": len(feature_ranking_df),
        "expected_value_base_spending_CZK": round(float(base_value), 4),
        "computation_time_seconds": round(float(computation_time_sec), 4),
        "feature_importance_ranking": feature_ranking_df.to_dict(orient="records"),
        "top_3_features": feature_ranking_df["feature"].head(3).tolist(),
        "archetype_explanations": archetypes_info,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return output_path
