# Strategy module description

A strategy module contains `trade-executor` logic needed to run a live trading strategy. It has corresponding sections for a Jupyter backtesting notebook it was created out of.

The strategy module starts with a module comment about what the strategy does.

Each strategy module have the sections described in this file. 

Sections are marked with a comment block like one below for `Charts` section.

```python
#
# Charts
#
```

## Strategy sections are

1. Imports
    - Contains Python imports
    - From notebooks, these are split across all cells, but in the strategy module they need to be at the top
2. Trading universe constants
    - Contains various contants used later in the module to choose chain, trading pairs and so on
    - In the notebook, this is `Chain configuration` section
3. Strategy parameters
    - Contains `Parameters` class
    - In the notebook, this is `Parameters` section
4. Strategy logic
    - Contains `decide_trades()`
    - In the notebook this is `Algorithm and backtest` section
5. Indicators
    - Contains `create_indicators()`
    - In the notebook this is `Indicators` section
6. Charts
    - Contains `create_charts()`
    - In the notebook this is `Charts set up` section
7. Metadata 
    - Contains constants: `tags`, `name`, `short_description`, `icon`, `long_description`
    - Notebook does not have this

## Examples

- Strategy module example: `strategies/master-vault.py`
- Notebook example: `.claude/skills/notebook-to-strategy/05-tweaked-basket-construction.ipynb`