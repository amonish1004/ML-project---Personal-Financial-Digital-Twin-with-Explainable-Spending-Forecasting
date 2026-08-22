import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Naive Persistence Baseline Model for monthly spending forecasting.
    
    Prediction rule:
    next_month_total_spending_hat(u, t+1) = spending_t(u)
    
    Non-parametric baseline establishing the minimum performance benchmark.
    Requires 0 learned parameters.
    """
    def __init__(self, target_lag_feature: str = "spending_t"):
        self.target_lag_feature = target_lag_feature

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "PersistenceBaseline":
        # Non-parametric baseline does not learn parameters
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.target_lag_feature not in X.columns:
                raise KeyError(f"Feature '{self.target_lag_feature}' not found in input DataFrame X.")
            return X[self.target_lag_feature].values.astype(float)
        elif isinstance(X, np.ndarray):
            # Assume first column (index 0) corresponds to spending_t
            return X[:, 0].astype(float)
        else:
            raise TypeError("Input X must be a pandas DataFrame or numpy ndarray.")


def build_ridge_pipeline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """
    Constructs a standardized Ridge Regression Pipeline.
    
    Pipeline Steps:
    1. StandardScaler: Fits mean and standard deviation strictly on the training partition.
    2. Ridge: Linear regression with L2 regularization penalty to prevent overfitting 
              and handle collinear lag features.
              
    Parameters
    ----------
    alpha : float, default=1.0
        L2 Regularization strength.
    random_state : int, default=42
        Random seed for reproducibility.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=random_state))
    ])
    return pipeline


def build_catboost_model(
    iterations: int = 300,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42
) -> CatBoostRegressor:
    """
    Constructs a non-linear CatBoostRegressor model.
    
    Parameters
    ----------
    iterations : int, default=300
        Number of decision trees.
    learning_rate : float, default=0.05
        Gradient step shrinkage size.
    depth : int, default=6
        Maximum depth of decision trees.
    random_seed : int, default=42
        Random seed for 100% reproducible tree building.
        
    Returns
    -------
    CatBoostRegressor
    """
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_seed,
        verbose=0
    )
    return model
