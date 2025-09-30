from __future__ import annotations
import numpy as np
import pandas as pd


def mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
	return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
	return float(np.mean(np.sign(y_true) == np.sign(y_pred)))
