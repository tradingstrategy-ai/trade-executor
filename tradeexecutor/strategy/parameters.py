"""Strategy input parameters handlding.

- Handle input parameters in single-run and grid search cases
"""
from web3.datastructures import ReadableAttributeDict


class StrategyParameters(ReadableAttributeDict):
    """Strategy parameters.

    These parameters may present

    - Individual constant parameters for the strategy, like indicators threshold levels:
      `rsi_low`, `rsi_high`

    - Parameters about the backtesting itself: `backtest_start`, `backtest_end`

    - Parameters about strategy execution: `cycle_duration`.

    The parameters are presented as attributed dict and
    are accessible using both dotted attribe access and dict access:

    .. code-block:: python

        assert parameters.rsi_low == parameters["rsi_low"]

    You can use `StrategyParameters` with inline backtest:

    .. code-block:: python

        def decide_trades(
                timestamp: pd.Timestamp,
                parameters: StrategyParameters,
                strategy_universe: TradingStrategyUniverse,
                state: State,
                pricing_model: PricingModel) -> List[TradeExecution]:
            assert parameters.test_val == 111
            # ...

        parameters = StrategyParameters({
            "test_val": 111,
        })

        # Run the test
        state, universe, debug_dump = run_backtest_inline(
            parameters=parameters,
        )

    When dealing with grid search input, every parameter is assumed to be a list
    of potential grid search combination values. `decide_trades` function
    gets called with every single combination of the list.

    .. code-block:: python

        from tradeexecutor.strategy.cycle import CycleDuration
        from pathlib import Path
        from tradeexecutor.backtest.grid_search import prepare_grid_combinations

        # This is the path where we keep the result files around
        storage_folder = Path("/tmp/v5-grid-search.ipynb")

        class StrategyParameters:
            cycle_duration = CycleDuration.cycle_1d
            rsi_days = [5, 6, 9, 8, 20]  # The length of RSI indicator
            eth_btc_rsi_days = 60  # The length of ETH/BTC RSI
            rsi_high = [60, 70, 80] # RSI trigger threshold for decision making
            rsi_low = [30, 40, 50]  # RSI trigger threshold for decision making
            allocation = 0.9 # Allocate 90% of cash to each position
            lookback_candles = 120
            minimum_rebalance_trade_threshold = 500.00  # Don't do trades that would have less than 500 USD value change

        combinations = prepare_grid_combinations(StrategyParameters, storage_folder)
        print(f"We prepared {len(combinations)} grid search combinations")

        # Then pass the combinations to a grid search
        from tradeexecutor.backtest.grid_search import perform_grid_search

        grid_search_results = perform_grid_search(
            decide_trades,
            strategy_universe,
            combinations,
            max_workers=8,
            trading_strategy_engine_version="0.4",
            multiprocess=True,
        )
    """


    @staticmethod
    def from_class(c: type, grid_search=False) -> "StrategyParameters":
        """Create parameter dict out from a class object.

        - Convert inlined class-style strategy parameter input to dictionary

        :param grid_search:
            Make grid-search style parameters.

            Every scalar input is converted to a single item list if set.
        """
        # https://stackoverflow.com/a/1939279/315168
        keys = [attr for attr in dir(c) if not callable(getattr(c, attr)) and not attr.startswith("__")]
        params = {k: getattr(c, k) for k in keys}

        if not grid_search:
            return StrategyParameters(params)

        # Convert single variable declarations to a list
        output = {}
        for k, v in params.items():
            if not type(v) in (list, tuple):
                v = [v]
            output[k] = v
        return StrategyParameters(output)