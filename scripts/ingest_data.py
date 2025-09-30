#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys

import pandas as pd

from qf603.data.loaders import load_csv_dataset
from qf603.data.schema import infer_schema


def main() -> int:
	parser = argparse.ArgumentParser(description="Ingest CSV to data/raw and emit schema.json")
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument("--out", default="data/raw", help="Destination folder for raw data")
	parser.add_argument("--timestamp", default="Date", help="Timestamp column to parse as datetime")
	args = parser.parse_args()

	src = Path(args.input)
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = load_csv_dataset(str(src), parse_dates=args.timestamp)
	print(f"loaded rows={len(df)} cols={df.shape[1]}")

	# Save a copy to raw
	dst = out_dir / src.name
	df.to_csv(dst, index=False)
	print(f"wrote raw copy: {dst}")

	# Infer and save schema
	schema = infer_schema(df)
	interim_dir = Path("data/interim")
	interim_dir.mkdir(parents=True, exist_ok=True)
	schema_path = interim_dir / "schema.json"
	with open(schema_path, "w", encoding="utf-8") as f:
		json.dump(schema.to_dict(), f, indent=2)
	print(f"wrote schema: {schema_path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
