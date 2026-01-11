# Notebook to strategy module skill

Transfer code from a backtesting Jupyter notebook to a Trade Executor strategy module.

## Inputs

- **Backtesting notebook**: A Jupyter notebook file (`.ipynb`) containing the strategy backtest
- **Strategy module**: A Python file (`.py`) that will be created or updated

Both of these follow similar, but not same, structure as defined in `.claude/skills/notebook-to-strategy/strategy-module-description.md`.

## How it works

The skill maps notebook sections to strategy module sections as defined in `strategy-module-description.md`.

## Process

1. **Read and parse** the notebook and existing strategy module (if any)
2. **Extract code** from notebook cells, mapping to the appropriate sections
3. **Consolidate imports** from all notebook cells to the top of the module
4. **Transfer functions and constants**:
   - `Parameters` class
   - `create_trading_universe()` function
   - `decide_trades()` function
   - Indicator functions decorated with `@indicators.define()`
   - `create_indicators()` wrapper function
   - Chart functions and `create_charts()` function
   - Module-level constants (CHAIN_ID, VAULTS, EXCHANGES, etc.)
5. **Preserve metadata** if updating an existing module, or create placeholder metadata for new modules
6. **Format the output** with proper section comments
7. **Verify** by running a backtest command

## Verification

After updating the strategy module, run a backtest to verify it works:

```shell
source .local-test.env && poetry run trade-executor \
    backtest \
    --strategy-file <strategy_module_path> 
```

Unless the script prints "All ok" at the end of it, it is a failure. 

- Either fix errors yourself
- Or if you do not understand the error, ask the user for help

## Display results to the user

After running the backtest, the CLI command outputs three generated files like:

- Notebook: /Users/moo/code/trade-executor/state/master-vault-backtest.ipynb
 HTML: /Users/moo/code/trade-executor/state/master-vault-backtest.html
- CSV: /Users/moo/code/trade-executor/state/master-vault-daily-returns.csv

From the HTML file, extract `Performance and risk metrics¶` table, format is so that you can display it inline in the chat, and print back test results of top 15 rows. 

## Notes

- The skill will preserve any existing metadata in the strategy module
- Imports are deduplicated and sorted
- Notebook-specific code (like `client = Client.create_jupyter_client()`) is excluded
- Backtest execution code (`run_backtest_inline()`) is excluded from the strategy module
- Chart rendering setup code specific to notebooks is excluded
- The `trading_strategy_engine_version` constant is preserved or set appropriately.
  Use the default "0.5"