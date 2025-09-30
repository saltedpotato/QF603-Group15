# Benchmark Spec (from paper, simplified)

- Target: next-day log return.
- Features: recent log return and rolling mean/std windows (5,20) as minimal proxy.
- Model: Ridge regression (paper’s linear regularized core).
- Train/test split: train <= 2022-12-31, test >= 2023-01-01.
- Decision: sign(pred) long/flat; costs 5 bps.

Note: Replace this with the exact paper method once extracted; this file serves as a placeholder for mapping choices 1:1 to our data.
