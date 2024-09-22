"""Size risk management to avoid too large trades and positions.

- See :py:class:`SizeRiskModel` for usage
"""

import abc

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.size_risk import SizingType, SizeRisk
from tradeexecutor.state.types import USDollarAmount, AnyTimestamp


class SizeRiskModel(abc.ABC):
    """Estimate an impact of a single trade.

    - We are going to take a price impact hit when taking liquidity out of the market

    - Handle max sizes for individual trades and positions,
      so that we do not create trades or positions that are too big
      compared to the available lit liquidity

    This is an abstract base class. See implementations for details.

    - capped fixed amount (no data needed)
    - historical real data (EVM archive node),
    - historical estimation (based on TVL)
    - live real data (EVM node)

    Example:

    .. code-block:: python

        from tradeexecutor.strategy.tvl_size_risk import HistoricalUSDTVLSizeRiskModel

        def decide_trades(input: StrategyInput) -> list[TradeExecution]:
            position_manager = input.get_position_manager()
            pair = input.get_default_pair()
            cash = input.state.portfolio.get_cash()
            timestamp = input.timestamp

            size_risker = HistoricalUSDTVLSizeRiskModel(
                input.pricing_model,
                per_trade_cap=0.02,  # Cap trade to 2% of pool TVL
            )

            trades = []
            if not position_manager.is_any_open():
                # Ask the size risk model what kind of estimation they give for this pair
                # and then cap the trade size based on this
                size_risk = size_risker.get_acceptable_size_for_buy(timestamp, pair, cash)
                # We never enter 100% position with our cash,
                # as floating points do not work well with ERC-20 18 decimal accuracy
                # and all kind of problematic rounding errors would happen.
                position_size = min(cash * 0.99, size_risk.accepted_size)
                trades += position_manager.open_spot(
                    pair,
                    position_size
                )
            else:
                trades += position_manager.close_all()
            return trades
    """

    @abc.abstractmethod
    def get_acceptable_size_for_buy(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_size: USDollarAmount,
    ) -> SizeRisk:
        pass

    @abc.abstractmethod
    def get_acceptable_size_for_sell(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_quantity: USDollarAmount,
    ) -> SizeRisk:
        raise NotImplementedError()

    def get_acceptable_size_for_position(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_value: USDollarAmount,
    ) -> SizeRisk:
        """What this the maximum position amount.

        - Avoid exceed maximum position size

        - If the position size is exceeded start to reduce the position

        - Default to the size you would allow with the
        """
        size = self.get_acceptable_size_for_buy(timestamp, pair, asked_value)
        size.sizing_type = SizingType.hold
        return size
