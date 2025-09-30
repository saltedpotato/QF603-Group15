from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import json
import pandas as pd


@dataclass
class Schema:
	columns: Dict[str, str]

	def to_dict(self) -> Dict[str, Any]:
		return {"columns": self.columns}

	@classmethod
	def from_dataframe(cls, df: pd.DataFrame) -> "Schema":
		mapping = {col: str(df[col].dtype) for col in df.columns}
		return cls(columns=mapping)

	def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
		issues = {"missing_columns": [], "extra_columns": [], "type_mismatches": {}}
		for col in self.columns:
			if col not in df.columns:
				issues["missing_columns"].append(col)
		for col in df.columns:
			if col not in self.columns:
				issues["extra_columns"].append(col)
		for col, dtype in self.columns.items():
			if col in df.columns and str(df[col].dtype) != dtype:
				issues["type_mismatches"][col] = {"expected": dtype, "found": str(df[col].dtype)}
		return issues

	def to_json(self, path: str) -> None:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(self.to_dict(), f, indent=2)


def infer_schema(df: pd.DataFrame) -> Schema:
	return Schema.from_dataframe(df)


def validate_schema(df: pd.DataFrame, schema: Schema) -> Dict[str, Any]:
	return schema.validate(df)
