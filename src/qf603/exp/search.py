from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from qf603.models.core import build_model
from qf603.models.metrics import mse, mae, directional_accuracy


@dataclass
class SearchResult:
	alpha: float
	fold_metrics: List[float]
	mean_metric: float


def time_series_cv_scores(X: pd.DataFrame, y: pd.Series, alphas: List[float], k: int = 5, metric: str = "mse") -> List[SearchResult]:
	results: List[SearchResult] = []
	tscv = TimeSeriesSplit(n_splits=k)
	for alpha in alphas:
		fold_scores: List[float] = []
		for train_idx, val_idx in tscv.split(X):
			X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
			y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
			cfg = {"model": {"alpha": alpha}}
			model = build_model(cfg)
			model.fit(X_tr, y_tr)
			pred = model.predict(X_va)
			if metric == "mse":
				score = mse(y_va, pred)
			elif metric == "mae":
				score = mae(y_va, pred)
			elif metric == "directional_accuracy":
				score = 1.0 - directional_accuracy(y_va, pred)
			else:
				raise ValueError(f"Unknown metric {metric}")
			fold_scores.append(score)
		results.append(SearchResult(alpha=float(alpha), fold_metrics=fold_scores, mean_metric=float(np.mean(fold_scores))))
	return results
