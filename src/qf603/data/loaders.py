from __future__ import annotations
from typing import Optional
import pandas as pd


def load_csv_dataset(path: str, parse_dates: Optional[str] = None) -> pd.DataFrame:
	kwargs = {}
	if parse_dates:
		kwargs["parse_dates"] = [parse_dates]
	df = pd.read_csv(path, **kwargs)
	return df
