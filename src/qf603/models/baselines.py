from __future__ import annotations
import numpy as np
import pandas as pd


def baseline_mean_return(train_returns: pd.Series, horizon: int = 1) -> float:
	"""Naive mean-return predictor."""
	return float(train_returns.mean())


def predict_constant(n: int, value: float) -> np.ndarray:
	return np.full(n, value, dtype=float)


def baseline_sign_persistence(train_returns: pd.Series) -> int:
	"""Predict sign of last observed return."""
	last = float(np.sign(train_returns.iloc[-1])) if len(train_returns) else 0.0
	return int(last)


def predict_sign_constant(n: int, sign_value: int, magnitude: float = 0.001) -> np.ndarray:
	return np.full(n, sign_value * magnitude, dtype=float)
