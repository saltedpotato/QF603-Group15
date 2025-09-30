#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml
import pandas as pd

from qf603.features.baseline import add_baseline_features, choose_price_column
from qf603.exp.search import time_series_cv_scores


def main() -> int:
	p = argparse.ArgumentParser(description="Ridge alpha time-series CV search")
	p.add_argument("--search", default="configs/search_v1.yaml")
	p.add_argument("--clean", default="data/processed/clean.parquet")
	p.add_argument("--out", default="reports/search")
	args = p.parse_args()

	cfg = yaml.safe_load(Path(args.search).read_text())
	clean = pd.read_parquet(args.clean)
	price_col = cfg.get("features", {}).get("price_col", choose_price_column(clean))
	windows = tuple(cfg.get("features", {}).get("windows", [5, 20]))
	feat = add_baseline_features(clean, price_col=price_col, windows=windows)
	ts_col = cfg["split"]["timestamp_column"]
	w1, w2 = windows
	feat = feat[[ts_col, "log_return", f"roll_mean_{w1}", f"roll_std_{w1}", f"roll_mean_{w2}", f"roll_std_{w2}"]].copy()
	X, y = feat.drop(columns=[ts_col]), feat["log_return"].shift(-int(cfg["search"]["horizon"]))
	mask = X.notna().all(axis=1) & y.notna()
	X, y = X.loc[mask], y.loc[mask]

	results = time_series_cv_scores(X, y, alphas=[float(a) for a in cfg["search"]["alphas"]], k=int(cfg["search"]["folds"]), metric=str(cfg["metrics"]["metric"]))
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	rows = [{"alpha": r.alpha, "mean_metric": r.mean_metric, "folds": r.fold_metrics} for r in results]
	(out_dir / "trials.json").write_text(json.dumps(rows, indent=2))
	best = min(results, key=lambda r: r.mean_metric)
	(out_dir / "best.json").write_text(json.dumps({"alpha": best.alpha, "mean_metric": best.mean_metric}, indent=2))
	print(json.dumps({"alpha": best.alpha, "mean_metric": best.mean_metric}, indent=2))
	# save best config
	(Path("configs") / "model_best.yaml").write_text(yaml.safe_dump({
		"model": {"type": "ridge", "alpha": float(best.alpha)},
		"features": {"price_col": price_col, "windows": list(windows)},
		"split": cfg["split"],
		"metrics": {"transaction_cost_bps": 5},
	}, sort_keys=False))
	print("wrote configs/model_best.yaml")
	return 0


if __name__ == "__main__":
	sys.exit(main())
