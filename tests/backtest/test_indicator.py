from pandas_ta import rsi

from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def test_indicators_single():
    """Parallel creation of indicators using a single run backtest."""

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet):
        indicators.create("rsi", {"length": parameters.rsi_length})
        indicators.create("sma_long", {"length": parameters.sma_long})
        indicators.create("sma_short", {"length": parameters.sma_short})

    class Parameters:
        rsi_length=20
        sma_long=200
        sma_short=12






