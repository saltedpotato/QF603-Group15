#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

from qf603.data.cleaning import clean_ohlcv
from qf603.features.baseline import add_baseline_features


def main() -> int:
	p = argparse.ArgumentParser(description="Prepare cleaned data and baseline features")
	p.add_argument("--input", required=True, help="Path to input raw CSV")
	p.add_argument("--timestamp", default="Date", help="Timestamp column name")
	p.add_argument("--price", default="Close", help="Price column for returns")
	p.add_argument("--out", default="data/processed", help="Output directory")
	args = p.parse_args()

	src = Path(args.input)
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(src, parse_dates=[args.timestamp])
	print(f"loaded raw: rows={len(df)} cols={df.shape[1]}")

	clean = clean_ohlcv(df, timestamp_column=args.timestamp)
	clean_path = out_dir / "clean.parquet"
	clean.to_parquet(clean_path, index=False)
	print(f"wrote cleaned: {clean_path} rows={len(clean)} cols={clean.shape[1]}")

	features = add_baseline_features(clean, price_col=args.price)
	feat_path = out_dir / "features.parquet"
	features.to_parquet(feat_path, index=False)
	print(f"wrote features: {feat_path} rows={len(features)} cols={features.shape[1]}")
	print(features.head().to_string(index=False))
	return 0


if __name__ == "__main__":
	sys.exit(main())
