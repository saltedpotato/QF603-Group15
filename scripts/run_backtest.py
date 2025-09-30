#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from qf603.features.baseline import add_baseline_features, choose_price_column
from qf603.models.core import build_model, _make_supervised
from qf603.eval.backtest import run_backtest, BacktestConfig


def main() -> int:
	p = argparse.ArgumentParser(description="Train model and run simple backtest")
	p.add_argument("--config", default="configs/model_v1.yaml")
	p.add_argument("--clean", default="data/processed/clean.parquet")
	p.add_argument("--out", default="reports/backtest")
	args = p.parse_args()

	cfg = yaml.safe_load(Path(args.config).read_text())
	clean = pd.read_parquet(args.clean)
	price_col = cfg.get("features", {}).get("price_col", choose_price_column(clean))
	windows = tuple(cfg.get("features", {}).get("windows", [5, 20]))
	feat = add_baseline_features(clean, price_col=price_col, windows=windows)
	ts_col = cfg["split"]["timestamp_column"]
	w1, w2 = windows
	feat = feat[[ts_col, "log_return", f"roll_mean_{w1}", f"roll_std_{w1}", f"roll_mean_{w2}", f"roll_std_{w2}"]].copy()

	train_end = pd.Timestamp(cfg["split"]["train_end"])
	test_start = pd.Timestamp(cfg["split"]["test_start"])
	train = feat[feat[ts_col] <= train_end]
	test = feat[feat[ts_col] >= test_start]

	X_train, y_train = _make_supervised(train.drop(columns=[ts_col]), target_col="log_return", horizon=1)
	X_test, y_test = _make_supervised(test.drop(columns=[ts_col]), target_col="log_return", horizon=1)

	model = build_model(cfg)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)

	bt_input = pd.DataFrame({ts_col: test.loc[X_test.index, ts_col], "y_true": y_test, "y_pred": pred})
	res = run_backtest(bt_input, BacktestConfig(transaction_cost_bps=float(cfg["metrics"]["transaction_cost_bps"])) )

	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	res["timeseries"].to_csv(out_dir / "metrics.csv", index=False)
	(Path(out_dir) / "summary.json").write_text(json.dumps(res["summary"], indent=2))
	print(json.dumps(res["summary"], indent=2))

	# Plot equity
	plt.figure(figsize=(10,4))
	res["timeseries"].plot(x=ts_col, y="equity", title="Equity Curve")
	plt.tight_layout()
	plots_dir = out_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(plots_dir / "equity.png")
	print(f"saved {out_dir}/metrics.csv and {plots_dir}/equity.png")
	return 0


if __name__ == "__main__":
	sys.exit(main())
