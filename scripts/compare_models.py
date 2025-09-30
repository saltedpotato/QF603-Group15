#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml
import pandas as pd

from qf603.features.baseline import add_baseline_features, choose_price_column
from qf603.models.core import build_model, _make_supervised
from qf603.models.metrics import mse, mae, directional_accuracy
from qf603.eval.backtest import run_backtest, BacktestConfig
from qf603.models.baselines import baseline_mean_return, predict_constant, baseline_sign_persistence, predict_sign_constant


def evaluate(y_true: pd.Series, y_pred: pd.Series, ts: pd.Series, cost_bps: float) -> dict:
	bt_in = pd.DataFrame({"Date": ts, "y_true": y_true, "y_pred": y_pred})
	res = run_backtest(bt_in, BacktestConfig(transaction_cost_bps=cost_bps))
	return {
		"mse": mse(y_true, y_pred),
		"mae": mae(y_true, y_pred),
		"directional_accuracy": directional_accuracy(y_true, y_pred),
		"mean_return": res["summary"]["mean_return"],
		"sharpe": res["summary"]["sharpe"],
		"final_equity": res["summary"]["final_equity"],
	}


def main() -> int:
	p = argparse.ArgumentParser(description="Compare ridge vs naive baselines")
	p.add_argument("--config", default="configs/model_v1.yaml")
	p.add_argument("--clean", default="data/processed/clean.parquet")
	p.add_argument("--out", default="reports/compare/leaderboard.csv")
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
	# Align timestamps for test X rows
	ts_test = test.loc[X_test.index, ts_col]

	# Ridge
	ridge = build_model(cfg)
	ridge.fit(X_train, y_train)
	yard_ridge = ridge.predict(X_test)
	res_ridge = evaluate(y_test, yard_ridge, ts_test, float(cfg["metrics"]["transaction_cost_bps"]))
	res_ridge["model"] = "ridge_v1"

	# Mean-return baseline
	mu = baseline_mean_return(y_train)
	res_mean = evaluate(y_test, predict_constant(len(y_test), mu), ts_test, float(cfg["metrics"]["transaction_cost_bps"]))
	res_mean["model"] = "mean_return"

	# Sign-persistence baseline
	sign = baseline_sign_persistence(y_train)
	res_sign = evaluate(y_test, predict_sign_constant(len(y_test), sign), ts_test, float(cfg["metrics"]["transaction_cost_bps"]))
	res_sign["model"] = "sign_persistence"

	leaderboard = pd.DataFrame([res_ridge, res_mean, res_sign])
	leaderboard = leaderboard.sort_values(["sharpe", "final_equity"], ascending=[False, False])
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	leaderboard.to_csv(args.out, index=False)
	print(leaderboard.to_string(index=False))
	return 0


if __name__ == "__main__":
	sys.exit(main())
