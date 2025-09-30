# Project assumptions

- Data source: `data/TLT_2007-01-01_to_2025-08-30.csv` assumed OHLCV daily.
- Timestamp column is `Date` in UTC; market holidays produce gaps.
- Target for v1: `Close` ; features will use returns derived from OHLCV.
- Missing values: forward-fill then drop leading NaNs.
- Outliers: 1% winsorization for returns.
- Split: train up to 2022-12-31, test from 2023-01-01.
- Costs: 5 bps per trade included in backtests.
- Paper/proposal alignment: will adjust once `reports/method_spec.md` is drafted.
