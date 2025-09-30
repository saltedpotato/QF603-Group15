#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import yaml
import pandas as pd

from qf603.features.baseline import add_baseline_features, choose_price_column
from qf603.models.core import build_model, _make_supervised
from qf603.models.metrics import mse, mae, directional_accuracy
from qf603.eval.backtest import run_backtest, BacktestConfig


def eval_config(clean: pd.DataFrame, price_col: str, windows: tuple[int,int], cost_bps: float, split: dict) -> dict:
	feat = add_baseline_features(clean, price_col=price_col, windows=windows)
	ts_col = split["timestamp_column"]
	w1, w2 = windows
	feat = feat[[ts_col, "log_return", f"roll_mean_{w1}", f"roll_std_{w1}", f"roll_mean_{w2}", f"roll_std_{w2}"]].copy()
	train_end = pd.Timestamp(split["train_end"]) ; test_start = pd.Timestamp(split["test_start"])
	train = feat[feat[ts_col] <= train_end] ; test = feat[feat[ts_col] >= test_start]
	X_tr, y_tr = _make_supervised(train.drop(columns=[ts_col]), target_col="log_return", horizon=1)
	X_te, y_te = _make_supervised(test.drop(columns=[ts_col]), target_col="log_return", horizon=1)
	model = build_model({"model": {"alpha": 0.01}})
	model.fit(X_tr, y_tr)
	pred = model.predict(X_te)
	bt_in = pd.DataFrame({ts_col: test.loc[X_te.index, ts_col], "y_true": y_te, "y_pred": pred})
	res = run_backtest(bt_in, BacktestConfig(transaction_cost_bps=cost_bps))
	return {
		"windows": list(windows),
		"cost_bps": cost_bps,
		"mse": mse(y_te, pred),
		"mae": mae(y_te, pred),
		"directional_accuracy": directional_accuracy(y_te, pred),
		"mean_return": res["summary"]["mean_return"],
		"sharpe": res["summary"]["sharpe"],
		"final_equity": res["summary"]["final_equity"],
	}


def main() -> int:
	p = argparse.ArgumentParser(description="Robustness sweep over windows and costs")
	p.add_argument("--config", default="configs/model_best.yaml")
	p.add_argument("--clean", default="data/processed/clean.parquet")
	p.add_argument("--out", default="reports/robustness.csv")
	args = p.parse_args()

	cfg = yaml.safe_load(Path(args.config).read_text())
	clean = pd.read_parquet(args.clean)
	price_col = cfg.get("features", {}).get("price_col", choose_price_column(clean))
	split = cfg["split"]
	window_grid = [(5,20), (10,20), (10,50), (20,50)]
	cost_grid = [0, 5, 10]
	rows = []
	for w in window_grid:
		for c in cost_grid:
			rows.append(eval_config(clean, price_col, w, float(c), split))
	out = pd.DataFrame(rows)
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(args.out, index=False)
	print(out.sort_values(["sharpe","final_equity"], ascending=[False, False]).head(10).to_string(index=False))
	return 0


if __name__ == "__main__":
	sys.exit(main())
