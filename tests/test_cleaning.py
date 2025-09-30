import pandas as pd
from qf603.data.cleaning import clean_ohlcv


def test_clean_basic():
	df = pd.DataFrame({"Date": ["2024-01-02", "2024-01-01", "2024-01-01"], "Close": [101, 100, 100]})
	out = clean_ohlcv(df)
	assert list(out["Date"].dt.strftime("%Y-%m-%d").values) == ["2024-01-01", "2024-01-02"]
