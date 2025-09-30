import json
from pathlib import Path
import pandas as pd

from qf603.data.schema import infer_schema, Schema, validate_schema


def test_infer_and_validate(tmp_path: Path):
	csv = tmp_path / "demo.csv"
	csv.write_text("Date,Close\n2024-01-01,100\n")
	df = pd.read_csv(csv, parse_dates=["Date"])
	schema = infer_schema(df)
	s = schema.to_dict()
	assert "columns" in s and set(s["columns"]) == {"Date", "Close"}

	issues = validate_schema(df, schema)
	assert issues["missing_columns"] == []
	assert issues["extra_columns"] == []
	assert issues["type_mismatches"] == {}
