# Profiling notebooks and backtests

Methods and scripts used to profile notebook cell execution and backtest performance.

## Scripts

### `profile_backtest.py` — profile a single backtest tick loop

Runs a synthetic 215-cycle EMA crossover backtest with `cProfile` to find bottlenecks in the backtest tick loop (the `run_backtest_inline()` hot path).

```shell
poetry run python scripts/profile_backtest.py
```

Output: top functions by cumulative and self time. Use this to find per-tick overhead like redundant `perform_integrity_check()`, duplicate `sync_interests()`, or expensive `calculate_position_statistics()`.

### `profile_notebook_cells.py` — profile each notebook cell

Profiles each non-backtest cell of the v49 optimiser notebook to find import and setup bottlenecks. Cells profiled:

- Cell 1: `Client.create_jupyter_client()` + `setup_charting_and_output()`
- Cell 3: Parameter imports (skopt, sklearn)
- Cell 5: `create_binance_universe()` (data loading)
- Cell 7: Display candle info
- Cell 9: Indicator imports (pandas_ta)
- Cell 11: Algorithm imports (quantstats, seaborn)

```shell
poetry run python scripts/profile_notebook_cells.py
```

Output: per-cell cProfile report showing function call counts and timings.

## Profiling inside notebooks

### `%%time` — wall clock timing

Add `%%time` to the first line of any cell for basic wall clock timing:

```python
%%time
strategy_universe = create_binance_universe(...)
```

### `%%prun` — cProfile inside a cell

Use `%%prun` for detailed profiling of a single cell:

```python
%%prun -s cumulative -q -l 80 -T backtest.prof

result = run_backtest_inline(...)
```

Then print the saved profile:

```python
print(open('backtest.prof', 'r').read())
```

Flags:
- `-s cumulative` — sort by cumulative time
- `-q` — suppress pager
- `-l 80` — show top 80 functions
- `-T backtest.prof` — save to file

## What to look for

### Import overhead (cell 1)

Look for transitive import chains pulling in heavy libraries. Example finding: `pair.py` → `vault.py` → `vault_metrics.py` → `ffn` added ~3s of import time. Fixed with lazy imports.

### Per-tick duplication (backtest loop)

Functions called N times per tick when once would suffice:

| Pattern | Example |
|---------|---------|
| Called 2-3x per tick | `setup_routing()`, `perform_integrity_check()` |
| Called for every open position every tick | `calculate_position_statistics()` |
| Materialises full list just to count | `len(list(get_all_trades()))` |
| Re-parses version strings every call | `is_version_greater_or_equal_than()` |

### Network calls without caching

Example: `BinanceDownloader.fetch_assets()` hit the API ~3,500 times per universe load. Fixed by caching the exchange info response.

## Interpreting results

When profiling cells sequentially in a script, Python imports are loaded once and shared across cells. This means:
- The first cell that triggers an import chain gets charged for the full import time
- Subsequent cells using the same modules appear fast
- Compare total time across all cells, not individual cells in isolation
