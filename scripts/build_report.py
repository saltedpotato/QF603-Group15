#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import sys
import pandas as pd


def main() -> int:
	root = Path('.')
	final = root / 'reports' / 'final'
	final.mkdir(parents=True, exist_ok=True)

	best_metrics_path = root / 'models' / 'model_best' / 'metrics.json'
	bt_best_path = root / 'reports' / 'backtest_best' / 'summary.json'
	leaderboard_path = root / 'reports' / 'compare' / 'leaderboard.csv'
	robust_path = root / 'reports' / 'robustness.csv'

	best_metrics = json.loads(best_metrics_path.read_text()) if best_metrics_path.exists() else {}
	bt_best = json.loads(bt_best_path.read_text()) if bt_best_path.exists() else {}
	leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
	robust = pd.read_csv(robust_path) if robust_path.exists() else pd.DataFrame()

	report_md = final / 'SUMMARY.md'
	lines: list[str] = []
	lines.append('# Final Report Summary')
	if best_metrics:
		lines.append('## Best Model Metrics')
		for k, v in best_metrics.items():
			lines.append(f'- {k}: {v}')
	if bt_best:
		lines.append('## Backtest (Best) Summary')
		for k, v in bt_best.items():
			lines.append(f'- {k}: {v}')
	if not leaderboard.empty:
		lines.append('## Leaderboard (Top 3 by Sharpe)')
		lines.append(leaderboard.sort_values(["sharpe","final_equity"], ascending=[False, False]).head(3).to_string(index=False))
	if not robust.empty:
		lines.append('## Robustness (Top 5 by Sharpe)')
		lines.append(robust.sort_values(["sharpe","final_equity"], ascending=[False, False]).head(5).to_string(index=False))
	report_md.write_text('\n'.join(lines))
	print(f"wrote {report_md}")
	return 0


if __name__ == '__main__':
	sys.exit(main())
