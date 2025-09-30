from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _make_supervised(df: pd.DataFrame, target_col: str, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
	"""Create supervised X, y where y is target shifted -horizon.

	Drops any rows that contain NaNs in features or target after shifting to ensure
	compatibility with scikit-learn validators.
	"""
	X_all = df.copy()
	y_all = X_all[target_col].shift(-horizon)
	# Valid rows: features all finite and target notna
	feature_ok = X_all.notna().all(axis=1)
	target_ok = y_all.notna()
	mask = feature_ok & target_ok
	X = X_all.loc[mask]
	y = y_all.loc[mask]
	return X, y


@dataclass
class RidgeModel:
	alpha: float = 1.0
	model: Ridge | None = None

	def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
		self.model = Ridge(alpha=self.alpha)
		self.model.fit(X, y)

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		assert self.model is not None
		return self.model.predict(X)


def build_model(cfg: Dict[str, Any]) -> RidgeModel:
	alpha = float(cfg.get("model", {}).get("alpha", 1.0))
	return RidgeModel(alpha=alpha)
