#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from qf603.features.baseline import add_baseline_features, choose_price_column


def main() -> int:
	p = argparse.ArgumentParser(description="Run EDA on processed data and save figures")
	p.add_argument("--input", default="data/processed/clean.parquet")
	p.add_argument("--out", default="reports/figures")
	p.add_argument("--timestamp", default="Date")
	args = p.parse_args()

	df = pd.read_parquet(args.input)
	print(f"loaded processed: rows={len(df)} cols={df.shape[1]}")

	price_col = choose_price_column(df)
	feat = add_baseline_features(df, price_col=price_col)
	print(f"features rows={len(feat)} cols={feat.shape[1]}")

	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Price over time
	plt.figure(figsize=(10,4))
	df.plot(x=args.timestamp, y=price_col, legend=False)
	plt.title(f"{price_col} over time")
	plt.tight_layout()
	p1 = out_dir / "price_over_time.png"
	plt.savefig(p1)
	print(f"saved {p1}")

	# Returns distribution
	sns.histplot(feat["log_return"].dropna(), bins=50, kde=True)
	plt.title("Log return distribution")
	plt.tight_layout()
	p2 = out_dir / "returns_hist.png"
	plt.savefig(p2)
	print(f"saved {p2}")

	# Rolling volatility
	plt.figure(figsize=(10,4))
	feat.plot(x=args.timestamp, y=["roll_std_5","roll_std_20"], title="Rolling volatility (std of log-returns)")
	plt.tight_layout()
	p3 = out_dir / "rolling_vol.png"
	plt.savefig(p3)
	print(f"saved {p3}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
