"""TVL-based trade and position size risking."""
import abc
import datetime
import enum
from decimal import Decimal

import pandas as pd

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.types import USDollarAmount, TokenAmount, USDollarPrice, AnyTimestamp
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.size_risk_model import SizeRiskModel, SizingType
from tradingstrategy.liquidity import LiquidityDataUnavailable
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

    def get_pair_cap(self, pair: TradingPairIdentifier, sizing_type: SizingType) -> Percent:
        """Get cap for an individual trade for a pair.

        - Different pool types can have different caps because of CLMM has better
          capital efficiency
        """
        match sizing_type:
            case SizingType.buy | SizingType.sell:
                return self.per_trade_cap
            case SizingType.hold:
                return self.per_position_cap
            case _:
                raise NotImplementedError()

    def get_acceptable_size_for_buy(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_size: USDollarAmount,
    ) -> SizeRisk:
        tvl = self.get_tvl(timestamp, pair)
        cap_pct = self.get_pair_cap(pair, SizingType.buy)
        tvl_cap = tvl * cap_pct
        accepted_size = min(tvl_cap, asked_size)
        capped = bool(accepted_size == tvl_cap)
        diagnostics_data = {
            "tvl": tvl,
            "cap_pct": cap_pct,
        }
        return SizeRisk(
            timestamp=timestamp,
            sizing_type=SizingType.buy,
            pair=pair,
            path=[pair],
            asked_size=asked_size,
            accepted_size=accepted_size,
            capped=capped,
            diagnostics_data=diagnostics_data,
            tvl=tvl,
        )

    def get_acceptable_size_for_sell(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_quantity: TokenAmount,
    ) -> SizeRisk:
        assert isinstance(asked_quantity, Decimal)
        mid_price = self.pricing_model.get_mid_price(timestamp, pair)
        asked_value = asked_quantity * mid_price
        tvl = self.get_tvl(timestamp, pair)
        cap_pct = self.get_pair_cap(pair, SizingType.sell)
        tvl_cap = tvl * cap_pct
        max_value = min(tvl_cap, asked_value)
        capped = bool(max_value == self.per_trade_cap)
        accepted_quantity = Decimal(max_value / mid_price)
        diagnostics_data = {
            "tvl": tvl,
            "cap_pct": cap_pct,
        }
        return SizeRisk(
            timestamp=timestamp,
            sizing_type=SizingType.sell,
            pair=pair,
            path=[pair],
            asked_quantity=asked_quantity,
            accepted_quantity=accepted_quantity,
            asked_size=asked_value,
            accepted_size=max_value,
            capped=capped,
            diagnostics_data=diagnostics_data,
            tvl=tvl,
        )

    def get_acceptable_size_for_position(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier,
        asked_value: USDollarAmount,
    ) -> SizeRisk:
        tvl = self.get_tvl(timestamp, pair)
        cap_pct = self.get_pair_cap(pair, SizingType.hold)
        tvl_cap = tvl * cap_pct
        accepted_size = min(tvl_cap, asked_value)
        capped = bool(accepted_size == tvl_cap)
        diagnostics_data = {
            "tvl": tvl,
            "cap_pct": cap_pct,
        }
        return SizeRisk(
            timestamp=timestamp,
            sizing_type=SizingType.hold,
            pair=pair,
            path=[pair],
            asked_size=asked_value,
            accepted_size=accepted_size,
            capped=capped,
            diagnostics_data=diagnostics_data,
            tvl=tvl,
        )

    @abc.abstractmethod
    def get_tvl(self, timestamp: AnyTimestamp | None, pair: TradingPairIdentifier) -> USDollarAmount:
        """Read the TVL from the underlying pricing model."""


class USDTVLSizeRiskModel(BaseTVLSizeRiskModel):
    """Estimate the trade size based historical USD TVL values.

    - Fast as we have preprocessed data available

    - Some tokens may spoof this value and give unrealistic sizes
    """

    def __init__(
        self,
        pricing_model: PricingModel,
        per_trade_cap: Percent = UNLIMITED_CAP,
        per_position_cap: Percent = UNLIMITED_CAP,
        missing_tvl_placeholder_usd: USDollarAmount = None,
    ):
        """Create size-risk model.

        :param pricing_model:
            Pricing model is used to read TVL data (historical/real time)

        :param per_trade_cap:
            How many % of pool TVL on trade can be

        :param per_position_cap:
            How many % of pool TVL on trade can be

        :parma missing_tvl_placeholder_usd:
            If we do not have TVL data available, use this value as a fixed US value placeholder.

            E.g. set to `250_000` to assume all unknown pools to have 250k TVL at any point of time.
        """
        super().__init__(
            pricing_model,
            per_trade_cap,
            per_position_cap
        )
        self.missing_tvl_placeholder = missing_tvl_placeholder_usd

    def get_tvl(self, timestamp: AnyTimestamp, pair: TradingPairIdentifier) -> USDollarAmount:
        """Read the TVL from the underlying pricing model."""

        exc = None
        try:
            tvl = self.pricing_model.get_usd_tvl(timestamp, pair)
        except LiquidityDataUnavailable as e:
            if self.missing_tvl_placeholder:
                tvl = self.missing_tvl_placeholder
            else:
                tvl = None
                exc = e

        assert tvl is not None, \
            f"HistoricalUSDTVLSizeRiskModel.get_tvl(): Cannot read TVL value at {timestamp} for pair {pair}\n" \
            f"Does the universe have liquidity data set up?\n" \
            f"Pricing model is: {self.pricing_model}\n" \
            f"Exception was: {exc}\n"
        return tvl


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

