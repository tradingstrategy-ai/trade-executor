"""Fixed size and unlimited trade size risking."""
import abc
import datetime
import enum
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.types import USDollarAmount, TokenAmount, USDollarPrice
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.size_risk_model import SizeRiskModel, SizingType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.types import Percent

#: We assume we are too rich to trade 10M trades/positions
UNLIMITED_CAP: Percent = 1.0


class TVLMethod(enum.Enum):
    """What kind of TVL estimator we use."""
    historical_usd_tracked = "historical_usd_tracked"
    raw_chain_based = "raw_chain_based"


class BaseTVLSizeRiskModel(SizeRiskModel):
    """A trade sizer that uses % of TVL of the target pool as the cap.

    - Reads TVL estimation and uses that to set the maximum size for a trade

    - Used in backtesting - fast to test as we do not need to query the historical
      liquidity from the archive node to get the accurate price impact
    """

    def __init__(
        self,
        pricing_model: PricingModel,
        per_trade_cap: Percent = UNLIMITED_CAP,
        per_position_cap: Percent = UNLIMITED_CAP,
    ):
        """

        :param per_trade_cap:
            Maximum US dollar value of a single trade, or unlimited.
        """
        self.pricing_model = pricing_model
        self.per_trade_cap = per_trade_cap
        self.per_position_cap = per_position_cap

    def get_acceptable_size_for_buy(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_size: USDollarAmount,
    ) -> SizeRisk:
        accepted_size = min(self.per_trade_cap, asked_size)
        capped = accepted_size == self.per_trade_cap
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.buy,
            pair=pair,
            path=[pair],
            asked_size=asked_size,
            accepted_size=accepted_size,
            capped=capped,
        )

    def get_acceptable_size_for_sell(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_quantity: TokenAmount,
    ) -> SizeRisk:
        mid_price = self.pricing_model.get_mid_price(timestamp, pair)
        asked_value = asked_quantity * mid_price
        max_value = max(self.per_trade_cap, asked_value)
        capped = max_value == self.per_trade_cap
        accepted_quantity = Decimal(max_value / mid_price)
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.buy,
            pair=pair,
            path=[pair],
            asked_quantity=asked_quantity,
            accepted_quantity=accepted_quantity,
            capped=capped,
        )

    def get_acceptable_size_for_position(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
        asked_value: USDollarAmount,
    ) -> SizeRisk:
        accepted_size = min(self.per_position_cap, asked_value)
        capped = accepted_size == self.per_position_cap
        return SizeRisk(
            timestamp=timestamp,
            type=SizingType.hold,
            pair=pair,
            path=[pair],
            asked_size=asked_value,
            accepted_size=accepted_size,
            capped=capped,
        )

    @abc.abstractmethod
    def get_tvl(self, timestamp: datetime.datetime | None, pair: TradingPairIdentifier) -> USDollarAmount:
        """Read the TVL from the underlying pricing model."""


class HistoricalUSDTVLSizeRiskModel(BaseTVLSizeRiskModel):
    """Estimate the trade size based historical USD TVL values.

    - Fast as we have preprocessed data available

    - Some tokens may spoof this value and give unrealistic sizes
    """

    def __init__(
        self,
        pricing_model: PricingModel,
        per_trade_cap: Percent = UNLIMITED_CAP,
        per_position_cap: Percent = UNLIMITED_CAP,
    ):
        super().__init__(
            pricing_model,
            per_trade_cap,
            per_position_cap
        )

    def get_tvl(self, timestamp: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Read the TVL from the underlying pricing model."""
        return self.pricing_model.get_usd_tvl(timestamp, pair)


class QuoteTokenTVLSizeRiskModel(BaseTVLSizeRiskModel):
    """Estimate the trade size based on raw quote tokens.

    - Directly query onchain value for the available liquidity in quote token,
      but slow as we need to use onchain data source

    - Imppossible to spoof

    - May not give accurate values

    - Not finished
    """

    def __init__(
        self,
        pair_universe: PandasPairUniverse,
        routing_model: RoutingModel,
        pricing_model: PricingModel,
        per_trade_cap: Percent = UNLIMITED_CAP,
        per_position_cap: Percent = UNLIMITED_CAP,
    ):
        self.pair_universe = pair_universe
        self.routing_model = routing_model
        super().__init__(
            pricing_model,
            per_trade_cap,
            per_position_cap
        )

    def get_tvl(self, timestamp: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Read the TVL from the underlying pricing model."""
        leg1, leg2 = self.routing_model.route_pair(self.pair_universe, pair)
        if leg2:
            path = [leg1, leg2]
        else:
            path = [leg1]
        rate = self.get_usd_conversion_rate(timestamp, path)
        tvl = self.pricing_model.get_quote_token_tvl(timestamp, pair)
        return tvl * rate

    def get_usd_conversion_rate(self, timestamp: datetime.datetime, path: list[TradingPairIdentifier]) -> USDollarPrice:
        """For multi-legged trades, get the USD conversion rate of the last leg.

        - E.g. when trading USD->ETH->BTC get the USD/ETH price and then we get BTC/USD price by multiplying ETH/BTC price with this.
        """
        assert len(path) > 0
        assert path[0].quote.is_stablecoin(), f"The starting point is not a stablecoin: {path[0]}"
        match len(path):
            case 1:
                return 1.0
            case 2:
                return self.pricing_model.get_mid_price(timestamp, path[0])
            case _:
                raise NotImplementedError(f"Only three-leg trades supported: {path}")

