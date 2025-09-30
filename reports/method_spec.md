# Method Spec (v1)

- Target: next-day log return of `Price`.
- Features: current `log_return`, rolling means/stds (5, 20 days).
- Model: Ridge regression.
- Train/test split: train <= 2022-12-31, test >= 2023-01-01.
- Metrics: MSE, MAE, directional accuracy.
- Backtest proxy: threshold 0, long if prediction>0 else flat; cost 5 bps when switching.
