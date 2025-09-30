import pandas as pd
from qf603.features.baseline import add_baseline_features


def test_features_shape():
	df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=25, freq="D"), "Close": range(25)})
	out = add_baseline_features(df)
	assert {"log_return", "roll_mean_5", "roll_std_5", "roll_mean_20", "roll_std_20"}.issubset(out.columns)
	assert len(out) <= len(df)
