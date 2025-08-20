import logging
import datetime
import math
import warnings
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Literal

import pandas as pd

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel

from tradeexecutor.state.types import USDollarPrice, Percent, USDollarAmount, AnyTimestamp
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse, LiquidityDataUnavailable
from tradingstrategy.pair import PandasPairUniverse, PairNotFoundError
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


class BacktestPricing(PricingModel):
    """Look up the historical prices.

    - By default, assume we can get buy/sell at
      open price of the timestamp

    - Different pricing model can be used for rebalances (more coarse)
      and stop losses (more granular)

    - This is a simple model and does not use liquidity data
      for the price impact estimation

    - We provide `data_delay_tolerance` to deal with potential
      gaps in price data
    """

    def __init__(
            self,
            candle_universe: GroupedCandleUniverse,
            routing_model: BacktestRoutingModel,
            data_delay_tolerance=pd.Timedelta("2d"),
            candle_timepoint_kind="open",
            very_small_amount=Decimal("0.10"),
            time_bucket: Optional[TimeBucket] = None,
            allow_missing_fees=False,
            trading_fee_override: float | None=None,
            liquidity_universe: GroupedLiquidityUniverse | None = None,
            fixed_prices: dict[TradingPairIdentifier, float] | None = None,
            pairs: Optional[PandasPairUniverse] = None,
            three_leg_resolution=True,
        ):
        """

        :param candle_universe:
            Candles where our backtesing date comes from

        :param routing_model:
            How do we route trades between different pairs
            TODO: Now ignored

        :param data_delay_tolerance:
            How long time gaps we allow in the backtesting data
            before aborting the backtesting with an exception.
            This is an safety check for bad data.

            Sometimes there cannot be trades for days
            if the blockchain has been halted,
            and thus no price data available.

        :param candle_timepoint_kind:
            Do we use opening or closing price in backtesting

        :param very_small_amount:
            What kind o a test amount we do use for a trade
            when we do not know the actual size of the trade.

        :param time_bucket:
            The granularity of the price data.

            Currently used for diagnostics and debug only.

        :param allow_missing_fees:
            Allow trading pairs with missing fee information.

            All trading pairs should have good fee information by default,
            unless dealing with legacy tests.

        :param trading_fee_override:
            Override the trading fee with a custom fee.

            See :py:meth:`set_trading_fee_override`.

        :param liquidity_universe:
            Used in TVL based position size limit.

            See :py:mod:`tradeexecutor.strategy.tvl_size_risk`.

        :param fixed_prices:
            Fix price of an asset to a certain value to work around missing data.

            Use then we do not candle price data available for a pair.
            Mainly to work around vault unit testing data issues.

        :param three_leg_resolution:
            Do we attempt to resolve three-legged trades fee structure.

            Disable to run legacy unit tests.

        """

        # TODO: Remove later - now to support some old code111
        if isinstance(candle_universe, TradingStrategyUniverse):
            pairs = candle_universe.data_universe.pairs
            candle_universe = candle_universe.data_universe.candles

        assert isinstance(candle_universe, GroupedCandleUniverse), f"Got candles in wrong format: {candle_universe.__class__}"

        # BacktestRoutingModel or GenericRouting
        # assert isinstance(routing_model, BacktestRoutingModel), f"Routing model must be BacktestRoutingModel got {routing_model.__class__}"

        self.candle_universe = candle_universe
        self.very_small_amount = very_small_amount
        self.routing_model = routing_model
        self.candle_timepoint_kind = candle_timepoint_kind
        self.data_delay_tolerance = data_delay_tolerance
        self.time_bucket = time_bucket
        self.allow_missing_fees = allow_missing_fees
        self.trading_fee_override = trading_fee_override
        self.liquidity_universe = liquidity_universe
        self.three_leg_resolution = three_leg_resolution

        if fixed_prices:
            for k, v in fixed_prices.items():
                assert isinstance(k, TradingPairIdentifier)
                assert isinstance(v, float), f"Fixed price must be a float, got {v} for {k}"
        self.fixed_prices = fixed_prices or {}

        # This was late additio,
        self.pairs = pairs

        # assert not three_leg_resolution

    def __repr__(self):
        return f"<BacktestSimplePricingModel bucket: {self.time_bucket}, candles: {self.candle_universe}>"

    def get_pair_for_id(self, internal_id: int) -> Optional[TradingPairIdentifier]:
        """Look up a trading pair.

        Useful if a strategy is only dealing with pair integer ids.
        """
        warnings.warn("Do not use internal ids as they are not stable ids."
                      "Instead use chain id + address tuples")

        pair = self.universe.pairs.get_pair_by_id(internal_id)
        if not pair:
            return None
        return translate_trading_pair(pair)

    def check_supported_quote_token(self, pair: TradingPairIdentifier):
        assert pair.quote.address == self.routing_model.reserve_token_address, f"Quote token {self.routing_model.reserve_token_address} not supported for pair {pair}, pair tokens are {pair.base.address} - {pair.quote.address}"

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Optional[Decimal]
    ) -> TradePricing:

        assert pair is not None, "Pair missing"

        if quantity:
            assert quantity > 0, f"Cannot sell negative amounts: {quantity} {pair}"

        # TODO: Include price impact
        pair_id = pair.internal_id

        fixed_price = self.fixed_prices.get(pair)
        if fixed_price:
            return TradePricing(
                price=float(fixed_price),
                mid_price=float(fixed_price),
                lp_fee=[0],
                pair_fee=[0],
                market_feed_delay=None,
                side=False,
                path=[pair]
            )

        mid_price, delay = self.candle_universe.get_price_with_tolerance(
            pair_id,
            ts,
            tolerance=self.data_delay_tolerance,
            kind=self.candle_timepoint_kind,
            pair_name_hint=pair.get_ticker(),
        )

        fee_result = self.get_pair_fee(ts, pair, "sell", separate_tax=True)
        if fee_result is None:
            pair_fee = tax_percent = None
        else:
            pair_fee, tax_percent = fee_result

        if pair_fee is not None:
            reserve = float(quantity) * mid_price
            lp_fee = float(reserve) * pair_fee
            tax = float(reserve) * tax_percent

            # Move price below mid price
            price = mid_price * (1 - pair_fee)

            if not pair.is_vault():
                assert lp_fee > 0, f"After simulating SELL trade, got non-positive LP fee: {pair} {quantity}: ${lp_fee}.\n"\
                               f"Mid price: {mid_price}, quantity: {quantity}, reserve: {reserve} , pair fee: {pair_fee}\n" \

        else:
            # Fee information not available
            if (not self.allow_missing_fees) and (self.trading_fee_override is None):
                raise AssertionError(f"Pair lacks fee information: {pair}")

            price = mid_price
            lp_fee = None
            tax = None

        return TradePricing(
            price=float(price),
            mid_price=float(mid_price),
            lp_fee=lp_fee,
            pair_fee=pair_fee,
            market_feed_delay=delay.to_pytimedelta(),
            side=False,
            path=[pair],
            token_tax=tax,
            token_tax_percent=tax_percent,
        )

    def get_buy_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       reserve: Optional[Decimal]) -> TradePricing:
        """Get the price for a buy transaction."""

        assert reserve is not None and reserve > 0, f"Tried to make a buy price estimation for zero or negative reserve amount.\n" \
                                                    f"Got reserve: {reserve} \n" \
                                                    f"For a buy estimation, please fill in the allocated reserve amount for: \n" \
                                                    f"{pair}.\n"


        # TODO: Include price impact
        pair_id = pair.internal_id

        # Unit test path to override the price for testing
        fixed_price = self.fixed_prices.get(pair)
        if fixed_price:
            return TradePricing(
                price=float(fixed_price),
                mid_price=float(fixed_price),
                lp_fee=[None],
                pair_fee=[None],
                market_feed_delay=None,
                side=True,
                path=[pair]
            )

        mid_price, delay = self.candle_universe.get_price_with_tolerance(
            pair_id,
            ts,
            tolerance=self.data_delay_tolerance,
            kind=self.candle_timepoint_kind,
        )

        assert mid_price not in (0, math.nan), f"Got bad mid price: {mid_price}"

        fee_result = self.get_pair_fee(ts, pair, "buy", separate_tax=True)
        if fee_result is None:
            pair_fee = tax_percent = None
        else:
            pair_fee, tax_percent = fee_result

        if pair_fee is not None:
            lp_fee = float(reserve) * pair_fee
            tax = float(reserve) * tax_percent

            # Move price above mid price
            price = mid_price * (1 + pair_fee)

            if self.trading_fee_override is None:
                if not pair.is_vault():
                    # Vault fees are zero
                    assert lp_fee > 0, f"Got bad fee: {pair} {reserve}: {lp_fee}, trading fee override is: {self.trading_fee_override}"
        else:
            # Fee information not available
            if not self.allow_missing_fees:
                raise AssertionError(f"Pair lacks fee information: {pair}")
            lp_fee = None
            price = mid_price
            tax = None

        assert price not in (0, math.nan) and price > 0, f"Got bad price: {price}"

        return TradePricing(
            price=float(price),
            mid_price=float(mid_price),
            lp_fee=lp_fee,
            pair_fee=pair_fee,
            market_feed_delay=delay.to_pytimedelta(),
            side=True,
            path=[pair],
            token_tax_percent=tax_percent,
            token_tax=tax,
        )

    def get_mid_price(self,
                      ts: datetime.datetime,
                      pair: TradingPairIdentifier) -> USDollarPrice:
        """Get the mid price by the candle."""
        pair_id = pair.internal_id

        price, delay = self.candle_universe.get_price_with_tolerance(
            pair_id,
            ts,
            tolerance=self.data_delay_tolerance,
            kind=self.candle_timepoint_kind,
        )
        return float(price)

    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        assert isinstance(pair, TradingPairIdentifier)
        decimals = pair.base.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=ROUND_DOWN)

    def get_pair_fee(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        direction: Literal["buy", "sell"],
        separate_tax=False,
    ) -> Optional[Percent] | Optional[tuple[Percent, Percent]]:
        """Figure out the fee from a pair or a routing.

        - What is the total cost of trading with this pair
        - With three-legged trades we need to account both legs
        """

        if self.trading_fee_override is not None:
            return self.trading_fee_override

        # Multi routing hack
        routing_model = self.routing_model
        if isinstance(routing_model, GenericRouting):
            routing_model, protocol_config = routing_model.get_router(pair)

        # Three legged, count in the fee in the middle leg
        if self.three_leg_resolution and (pair.quote.address != routing_model.reserve_token_address.lower()):
            intermediate_pairs = routing_model.allowed_intermediary_pairs
            assert self.pairs is not None, "To do three-legged fee resolution, we need to get access to pairs in constructor"

            pair_address = intermediate_pairs.get(pair.quote.address)
            assert pair_address, f"No intermediate pair configured for {pair.quote} in {intermediate_pairs}, routing model is {routing_model}"
            try:
                extra_leg = self.pairs.get_pair_by_smart_contract(pair_address)
            except PairNotFoundError as e:
                raise RuntimeError(f"Trading Pair universe does not have pair {pair_address} for three-legged trade resolution. Allowed intermediary pairs: {intermediate_pairs}") from e
            extra_fee = extra_leg.fee_tier
        else:
            extra_fee = 0

        if direction == "buy":
            # Get token tax
            tax = pair.base.get_buy_tax() or 0
        elif direction == "sell":
            tax = pair.base.get_sell_tax() or 0
        else:
            raise NotImplementedError(f"Unsupported direction {direction} for pair {pair}")

        # Pair has fee information
        result = None
        if pair.fee is not None:
            result = pair.fee + extra_fee + tax
        else:

            # Pair does not have fee information, assume a default fee
            default_fee = self.routing_model.get_default_trading_fee()
            if default_fee:
                tax = 0
                result = default_fee + extra_fee + tax

        if result is not None:
            if separate_tax:
                return result, tax
            else:
                return result

        # None of pricing data available for this pair.
        # Legacy. Should not happen.
        return None

    def set_trading_fee_override(
            self,
            trading_fee_override: Percent | None
    ):
        self.trading_fee_override = trading_fee_override

    def get_usd_tvl(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        """Get the available liquidity at the opening of the day."""
        assert self.liquidity_universe is not None, "liquidity_universe not passed to BacktestPricing constructor"

        if isinstance(timestamp, datetime.datetime):
            timestamp = pd.Timestamp(timestamp)

        try:
            tvl, when = self.liquidity_universe.get_liquidity_with_tolerance(
                pair.internal_id,
                timestamp,
                tolerance=self.data_delay_tolerance,
                kind="open"
            )
        except LiquidityDataUnavailable as e:
            # Show the pair naem
            raise LiquidityDataUnavailable(f"Could not read TVL/liquidity data for {pair} - see nested exception for details") from e

        assert tvl is not None, "get_liquidity_with_tolerance() returned None: likely cause is that synthetic backtest data period mismatches backtest"

        return tvl


def backtest_pricing_factory(
        execution_model: ExecutionModel,
        universe: TradingStrategyUniverse,
        routing_model: UniswapV2Routing) -> BacktestPricing:

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, BacktestExecution), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, (BacktestRoutingModel, UniswapV2Routing)), f"This pricing method only works with Uniswap routing model, we received {routing_model}"

    return BacktestPricing(
        universe.data_universe.candles,
        routing_model,
        pairs=universe.data_universe.pairs,
    )

