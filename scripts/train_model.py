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


def main() -> int:
	p = argparse.ArgumentParser(description="Train v1 model on processed features")
	p.add_argument("--config", default="configs/model_v1.yaml")
	p.add_argument("--clean", default="data/processed/clean.parquet")
	p.add_argument("--out", default="models/model_v1")
	args = p.parse_args()

	cfg = yaml.safe_load(Path(args.config).read_text())
	clean = pd.read_parquet(args.clean)
	price_col = cfg.get("features", {}).get("price_col", choose_price_column(clean))
	windows = tuple(cfg.get("features", {}).get("windows", [5, 20]))
	feat = add_baseline_features(clean, price_col=price_col, windows=windows)
	# Keep timestamp column for splitting
	ts_col = cfg["split"]["timestamp_column"]
	w1, w2 = windows
	feat = feat[[ts_col, "log_return", f"roll_mean_{w1}", f"roll_std_{w1}", f"roll_mean_{w2}", f"roll_std_{w2}"]].copy()

	train_end = pd.Timestamp(cfg["split"]["train_end"])
	test_start = pd.Timestamp(cfg["split"]["test_start"])

	# Guard: make sure timestamp column is present
	if ts_col not in feat.columns:
		raise KeyError(f"Timestamp column '{ts_col}' not found in features. Available: {list(feat.columns)}")

	train = feat[feat[ts_col] <= train_end]
	test = feat[feat[ts_col] >= test_start]

	X_train, y_train = _make_supervised(train.drop(columns=[ts_col]), target_col="log_return", horizon=1)
	X_test, y_test = _make_supervised(test.drop(columns=[ts_col]), target_col="log_return", horizon=1)

	model = build_model(cfg)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)

	metrics = {"mse": mse(y_test, pred), "mae": mae(y_test, pred), "directional_accuracy": directional_accuracy(y_test, pred)}
	print("metrics", json.dumps(metrics, indent=2))

	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	(Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
	print(f"saved metrics to {out_dir}/metrics.json")
	return 0


if __name__ == "__main__":
	sys.exit(main())
