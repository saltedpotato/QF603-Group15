from __future__ import annotations
import pandas as pd
import numpy as np


def _normalized_name(name: str) -> str:
	return "".join(ch.lower() for ch in name if ch.isalnum())


def choose_price_column(df: pd.DataFrame, preferred: str = "Close") -> str:
	"""Return an existing price column name from the dataframe.

	Tries preferred first; falls back to common alternatives like Adj Close.
	Raises ValueError if none found.
	"""
	if preferred in df.columns:
		return preferred

	# Build case/format-insensitive lookup
	normalized_to_original = {_normalized_name(c): c for c in df.columns}
	candidates = [
		preferred,
		"Adj Close",
		"Adj_Close",
		"AdjClose",
		"close",
		"adj_close",
		"adjclose",
		"Price",
		"price",
	]
	for cand in candidates:
		key = _normalized_name(cand)
		if key in normalized_to_original:
			return normalized_to_original[key]
	raise ValueError(
		f"No suitable price column found. Tried: {candidates}. Available: {list(df.columns)}"
	)


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
	# Resolve the actual price column to use
	actual_price_col = choose_price_column(df, preferred=price_col)
	px = df[actual_price_col].astype(float)
	ret = np.log(px).diff()
	return ret.rename("log_return")


def add_baseline_features(
	df: pd.DataFrame,
	price_col: str = "Close",
	windows: tuple[int, int] = (5, 20),
) -> pd.DataFrame:
	out = df.copy()
	ret = compute_log_returns(out, price_col)
	out["log_return"] = ret
	w1, w2 = windows
	out[f"roll_mean_{w1}"] = ret.rolling(w1).mean()
	out[f"roll_std_{w1}"] = ret.rolling(w1).std(ddof=0)
	out[f"roll_mean_{w2}"] = ret.rolling(w2).mean()
	out[f"roll_std_{w2}"] = ret.rolling(w2).std(ddof=0)
	out = out.dropna()
	return out
