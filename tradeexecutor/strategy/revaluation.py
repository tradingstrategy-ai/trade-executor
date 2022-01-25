import pandas as pd

from tradeexecutor.state.state import TradingPosition
from tradingstrategy.universe import Universe


class RevaluationFailed(Exception):
    """Should not happen."""


class ClosePriceRevaluator:
    """Re-value assets based on their closing price at the timestamp.

    Used to calculate the portfolio total equity.

    The assumption is following

    - A data feed produces a new daily candle at 00:00
    - A notification hook is called
    - The strategy tick() kicks off
    - As the first step of the strategy, all current open positions are revalued
    - The revaluation happens on the closing price of the previous day
    """

    def __init__(self, universe: Universe, lookback_candles=5):
        self.universe = universe
        self.lookback_candles = lookback_candles

    def __call__(self, timestamp: pd.Timestamp, position: TradingPosition):
        """
        :param timestamp:
        :param position:
        :raise RevaluationFailed:
        :raise PriceUnavailable:
        """
        pair = position.pair.get_trading_pair(self.universe.pairs)
        if not pair:
            raise RevaluationFailed(f"The trading pair {position} has disappeared from the pair universe")

        price = self.universe.candles.get_closest_price(pair.pair_id, timestamp, self.lookback_candles)
        return price





