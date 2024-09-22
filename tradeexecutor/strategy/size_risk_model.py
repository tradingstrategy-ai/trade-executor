"""Size risk management to avoid too large trades and positions."""

import abc

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.size_risk import SizingType, SizeRisk
from tradeexecutor.state.types import USDollarAmount, AnyTimestamp


class SizeRiskModel(abc.ABC):
    """Estimate an impact of a single trade.

    - We are going to take a hit when taking liquidity out of the market

    - Handle max sizes for individual trades and positions

    Estimate this based on either

    - capped fixed amount (no data needed)
    - historical real data (EVM archive node),
    - historical estimation (based on TVL)
    - live real data (EVM node)
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
        size.type = SizingType.hold
        return size
