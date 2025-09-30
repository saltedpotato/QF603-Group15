from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
	transaction_cost_bps: float = 5.0
	threshold: float = 0.0


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, Any]:
	# df must contain columns: Date, y_true (next-day return), y_pred
	out = df.copy().reset_index(drop=True)
	# Position: 1 if pred>threshold else 0 (long/flat)
	out["position"] = (out["y_pred"] > cfg.threshold).astype(int)
	# Transaction when position changes
	out["trade"] = out["position"].diff().abs().fillna(0)
	cost = cfg.transaction_cost_bps / 1e4
	# Strategy return: position * true_return - cost*|delta position|
	out["strategy_ret"] = out["position"] * out["y_true"] - cost * out["trade"]
	out["equity"] = (1 + out["strategy_ret"]).cumprod()
	metrics = {
		"mean_return": float(out["strategy_ret"].mean()),
		"vol": float(out["strategy_ret"].std(ddof=0)),
		"sharpe": float(out["strategy_ret"].mean() / (out["strategy_ret"].std(ddof=0) + 1e-12)),
		"final_equity": float(out["equity"].iloc[-1]) if len(out) else 1.0,
	}
	return {"timeseries": out, "summary": metrics}
