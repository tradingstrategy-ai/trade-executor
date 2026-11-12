# Quarterly and annual profit comparisons

Compare strategies with their baselines quarter by quarter and year by year,
alongside aggregate CAGR, Sharpe and continuous drawdown. A quarterly heatmap
uses years on rows, Q1–Q4 and observed Year totals on columns. Each cell shows
percentage return on opening equity; dollar amounts are not displayed. Year totals
cover observed support and are not annualised; percentage returns compound.

```python
from tradeexecutor.visual.calendar import calculate_calendar_profit, visualise_quarterly_profit

table = calculate_calendar_profit(
    equity, initial_cash=100_000,
    start_at="2025-04-01", end_at="2026-10-21",
)
figure = visualise_quarterly_profit(table, title="Baseline: quarterly returns")
figure.write_image("quarterly-profit.png")
table.to_csv("quarterly-profit.csv")
```

The UTC evaluation interval is start inclusive, end exclusive. Valuations are
daily by default; pass the actual `valuation_interval` for intraday backtests.
Each timestamp books the change from the preceding valuation. The first
valuation uses the initial deposit as its opening equity, retaining first-day
fees and losses. USD quarterly profits telescope to final equity minus initial
cash; returns compound to the total equity return. This is equity profit,
including unrealised gains and recorded costs, not realised closed-trade PnL.

The fixed-capital curve must contain no external deposits or withdrawals.
Live fund flows need separate adjustment; the registered chart refuses live
mode. Missing, duplicated, unordered, non-finite and non-positive valuations
fail explicitly. No missing observations are replaced with zero profit or
forward-filled here. Carried or synthetic underlying asset NAVs still need
qualification by the caller. Empty curves produce an explicit no-data figure.

Unobserved quarters are grey N/A cells. Asterisks mark calendar quarters and
years clipped by the evaluation interval. A partial year is not a full-year
result. Colours encode percentage return on a symmetric scale centred at zero;
use a common `colour_limit` for compared wallets, expressed as a fractional
return (0.10 means 10%). Cells, hover text and colour-bar ticks use percentages
only. All chart text uses 16 px Arial. Long titles wrap above a separate UTC
calendar subtitle; quarter labels and explanatory footnotes have their own space. Full-precision tables retain UTC
support boundaries and observation counts. Inspect exported cell labels at
comment width, and retain weak periods rather than selecting favourable ones.

The stock `tradeexecutor.strategy.chart.standard.calendar.quarterly_profit`
chart registers as `ChartKind.state_all_pairs` and returns `(figure, table)`.
It uses the backtest state's initial cash and the chart input's evaluation
boundaries. This API complements the existing monthly return chart without
changing it. Reporting from saved equity or state does not rerun the strategy.

The layout was informed by the Q1–Q4/Year comparison in the
[research chart catalogue](https://github.com/tradingstrategy-ai/freqtrade-strategies/blob/master/.claude/docs/charts.md)
and its `plot_quarterly_heatmap` helper. The accounting here additionally retains
initial-day costs, explicit valuation coverage and partial-year labels.
