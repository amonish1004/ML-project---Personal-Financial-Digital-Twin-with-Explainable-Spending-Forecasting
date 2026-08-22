import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

PREDICTOR_FEATURES = [
    "spending_t",
    "spending_t_minus_1",
    "spending_t_minus_2",
    "spending_3m_mean",
    "spending_3m_std",
    "debit_count_t",
    "income_credit_t",
    "ending_balance_t",
    "spending_hh_t",
    "spending_st_t",
    "spending_in_t",
    "spending_lo_t",
    "spending_io_t",
    "spending_other_t",
]

TARGET_COLUMN = "next_month_total_spending"

TRAIN_TARGET_MONTH_BOUNDS = ("2013-04", "2016-12")
VAL_TARGET_MONTH_BOUNDS = ("2017-01", "2017-12")
TEST_TARGET_MONTH_BOUNDS = ("2018-01", "2018-12")


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard continuous regression evaluation metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    - MedAE (Median Absolute Error)
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MedAE": round(medae, 4),
    }


def load_and_split_dataset(csv_path: Path) -> Dict[str, Any]:
    """
    Loads the supervised dataset and performs a strict out-of-time temporal partition based on target_month.

    Partitions:
    - TRAIN: target_month 2013-04 through 2016-12 (Expected: 71,824 samples)
    - VALIDATION: target_month 2017-01 through 2017-12 (Expected: 45,980 samples)
    - TEST: target_month 2018-01 through 2018-12 (Expected: 53,390 samples)

    Safety checks:
    - 0 row drops / 100% total row accounting
    - No random shuffling
    - Predictor matrix X includes only the 14 verified predictor features
    - Target vector y contains only next_month_total_spending
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Supervised dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate presence of expected columns
    for col in PREDICTOR_FEATURES + [TARGET_COLUMN, "target_month", "account_id"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from supervised dataset.")

    # Chronological Out-of-Time Partitioning
    df_train = df[
        (df["target_month"] >= TRAIN_TARGET_MONTH_BOUNDS[0]) &
        (df["target_month"] <= TRAIN_TARGET_MONTH_BOUNDS[1])
    ].copy()

    df_val = df[
        (df["target_month"] >= VAL_TARGET_MONTH_BOUNDS[0]) &
        (df["target_month"] <= VAL_TARGET_MONTH_BOUNDS[1])
    ].copy()

    df_test = df[
        (df["target_month"] >= TEST_TARGET_MONTH_BOUNDS[0]) &
        (df["target_month"] <= TEST_TARGET_MONTH_BOUNDS[1])
    ].copy()

    # Safety checks
    total_split_len = len(df_train) + len(df_val) + len(df_test)
    if total_split_len != len(df):
        raise ValueError(
            f"Temporal split partition error: Total split rows ({total_split_len:,}) "
            f"does not match total dataset rows ({len(df):,})."
        )

    # Feature and Target extraction
    X_train = df_train[PREDICTOR_FEATURES].copy()
    y_train = df_train[TARGET_COLUMN].values

    X_val = df_val[PREDICTOR_FEATURES].copy()
    y_val = df_val[TARGET_COLUMN].values

    X_test = df_test[PREDICTOR_FEATURES].copy()
    y_test = df_test[TARGET_COLUMN].values

    split_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "df_train_meta": df_train[["account_id", "reference_month", "target_month"]],
        "df_val_meta": df_val[["account_id", "reference_month", "target_month"]],
        "df_test_meta": df_test[["account_id", "reference_month", "target_month"]],
        "metadata": {
            "total_samples": len(df),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "train_accounts": df_train["account_id"].nunique(),
            "val_accounts": df_val["account_id"].nunique(),
            "test_accounts": df_test["account_id"].nunique(),
            "train_period": f"{df_train['target_month'].min()} to {df_train['target_month'].max()}",
            "val_period": f"{df_val['target_month'].min()} to {df_val['target_month'].max()}",
            "test_period": f"{df_test['target_month'].min()} to {df_test['target_month'].max()}",
        },
    }

    return split_data
