from __future__ import annotations
from typing import Tuple
import pandas as pd


def coerce_types(df: pd.DataFrame, timestamp_column: str = "Date") -> pd.DataFrame:
	out = df.copy()
	if timestamp_column in out.columns:
		out[timestamp_column] = pd.to_datetime(out[timestamp_column], errors="coerce")
	return out


def sort_and_deduplicate(df: pd.DataFrame, timestamp_column: str = "Date") -> pd.DataFrame:
	out = df.sort_values(timestamp_column).drop_duplicates(subset=[timestamp_column])
	return out.reset_index(drop=True)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out = out.ffill()  # forward fill
	out = out.dropna() # drop any remaining
	return out


def clean_ohlcv(
	df: pd.DataFrame,
	timestamp_column: str = "Date",
) -> pd.DataFrame:
	out = coerce_types(df, timestamp_column)
	out = sort_and_deduplicate(out, timestamp_column)
	out = fill_missing(out)
	return out
