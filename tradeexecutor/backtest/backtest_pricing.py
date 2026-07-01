import datetime
import logging
import math
import warnings
from decimal import ROUND_DOWN, Decimal
from typing import Literal, Optional

import numpy as np
import pandas as pd
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.vault_windows import VaultWindowSchedule
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import \
    UniswapV2Routing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import (AnyTimestamp, Percent, USDollarAmount,
                                       USDollarPrice)
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.redemption import (
    RedemptionBlockReason,
    RedemptionCheckResult,
    RedemptionCheckStage,
)
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, translate_trading_pair)
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.liquidity import (GroupedLiquidityUniverse,
                                       LiquidityDataUnavailable)
from tradingstrategy.pair import PairNotFoundError, PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket

logger = logging.getLogger(__name__)


def _bool_series_to_sentinel(series: pd.Series) -> np.ndarray:
    """Encode a nullable-boolean availability column as int8 sentinels.

    ``-1`` unknown / NA, ``0`` closed (False), ``1`` open (True). Keeps the fast numpy
    searchsorted lookup path free of pandas nullable/object dtypes.
    """
    return series.astype("boolean").map({True: 1, False: 0}).fillna(-1).to_numpy(dtype=np.int8)


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
        ignore_routing=False,
        vault_state: pd.DataFrame | None = None,
        vault_window_overrides: dict[int, VaultWindowSchedule] | None = None,
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

        :param ignore_routing:
            Ignore three-leg routing and assume three-leg swap fee is zero.

            Currently needed for cross-chain backtesting.

        :param vault_state:
            Per-(vault, timestamp) deposit/redemption availability state.

            Tidy DataFrame from
            :py:func:`tradingstrategy.alternative_data.vault.convert_vault_prices_to_vault_state`
            (``pair_id``, ``timestamp`` plus available ``VAULT_STATE_COLUMNS``). When provided,
            :py:meth:`can_deposit` / :py:meth:`check_redemption` answer from this history so
            backtests skip deposits into / sells out of a vault that was closed at that timestamp.
            This models open/closed availability only, not partial size caps. Unknown / missing /
            out-of-tolerance values are treated as "allowed".

        """

        # TODO: Remove later - now to support some old code111
        if isinstance(candle_universe, TradingStrategyUniverse):
            pairs = candle_universe.data_universe.pairs
            # Default vault-state from the universe so the shortcut path keeps deposit/redemption
            # gating even when callers do not pass `vault_state` explicitly.
            if vault_state is None:
                vault_state = candle_universe.vault_state
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
        self.ignore_routing = ignore_routing
        # Synthetic per-vault deposit/redemption window schedules (backtest modelling).
        # Consulted BEFORE the real vault_state, so an override beats stale/always-open data.
        self._vault_window_overrides: dict[int, VaultWindowSchedule] = vault_window_overrides or {}

        if fixed_prices:
            for k, v in fixed_prices.items():
                assert isinstance(k, TradingPairIdentifier)
                assert isinstance(v, float), f"Fixed price must be a float, got {v} for {k}"
        self.fixed_prices = fixed_prices or {}

        # This was late additio,
        self.pairs = pairs

        # Pre-index liquidity data for fast TVL lookups.
        #
        # get_usd_tvl() is called ~23 times per tick (once per signal pair)
        # during normalise_weights(). The standard path through
        # get_liquidity_with_tolerance() costs ~0.5ms per call due to pandas
        # overhead (isinstance checks, get_indexer, iloc). Pre-building numpy
        # arrays reduces each call to ~1µs via numpy.searchsorted.
        #
        # Each entry: pair_id → (sorted_timestamps_seconds, open_values)
        # where timestamps are int64 UNIX seconds for fast comparison.
        # Memory: ~120 pairs × 1000 candles × 16 bytes ≈ 2MB.
        self._tvl_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if self.liquidity_universe is not None:
            for pair_id in self.liquidity_universe.get_pair_ids():
                try:
                    samples = self.liquidity_universe.get_samples_by_pair(pair_id)
                except KeyError:
                    continue
                if isinstance(samples.index, pd.MultiIndex):
                    ts_values = samples.index.get_level_values(1).values
                else:
                    ts_values = samples.index.values
                # Store as int64 UNIX seconds for fast numpy.searchsorted
                self._tvl_arrays[pair_id] = (
                    ts_values.astype("datetime64[s]").astype("int64"),
                    samples["open"].values.astype(float),
                )

        # Pre-index per-(vault, timestamp) deposit/redemption availability state the same way,
        # so can_deposit()/check_redemption() are O(log K) searchsorted lookups in the hot path.
        #
        # Booleans are stored as compact int8 sentinels: -1 unknown, 0 closed, 1 open. Caps are
        # float arrays (NaN = unknown). Reasons are kept as object arrays for diagnostics.
        # Each entry: pair_id → {"ts": int64 seconds, "deposits_open": int8, ...}
        self.vault_state = vault_state
        self._vault_state_arrays: dict[int, dict[str, np.ndarray]] = {}
        if vault_state is not None and len(vault_state) > 0:
            for pair_id, group in vault_state.groupby("pair_id", sort=False):
                group = group.sort_values("timestamp")
                entry: dict[str, np.ndarray] = {
                    "ts": group["timestamp"].values.astype("datetime64[s]").astype("int64"),
                }
                for col in ("deposits_open", "redemption_open"):
                    if col in group.columns:
                        entry[col] = _bool_series_to_sentinel(group[col])
                for col in ("max_deposit", "max_redeem"):
                    if col in group.columns:
                        entry[col] = group[col].to_numpy(dtype=float)
                for col in ("deposit_closed_reason", "redemption_closed_reason"):
                    if col in group.columns:
                        entry[col] = group[col].to_numpy(dtype=object)
                self._vault_state_arrays[int(pair_id)] = entry

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
            if separate_tax:
                return self.trading_fee_override, 0
            else:
                return self.trading_fee_override

        # Multi routing hack
        routing_model = self.routing_model
        if isinstance(routing_model, GenericRouting):
            routing_model, protocol_config = routing_model.get_router(pair)

        # Three legged, count in the fee in the middle leg
        if self.three_leg_resolution and (pair.quote.address != routing_model.reserve_token_address.lower()) and not self.ignore_routing:
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
        pair: TradingPairIdentifier,
    ) -> USDollarAmount:
        """Get the available liquidity for a pair at a given timestamp.

        Uses pre-indexed numpy arrays for O(log K) lookups without
        pandas overhead. Semantically identical to
        ``liquidity_universe.get_liquidity_with_tolerance()``:
        backward-fill (most recent value at or before timestamp)
        with tolerance validation.

        The pre-indexing happens at ``__init__()`` time, converting each
        pair's liquidity data from a pandas DataFrame into sorted numpy
        arrays. Each lookup is a single ``numpy.searchsorted()``
        call (~1µs) instead of the full pandas path (~0.5ms).

        Profiling showed the original path consuming ~13s across a
        6-backtest optimiser run (25,625 calls × 0.5ms each).
        """
        assert self.liquidity_universe is not None, \
            "liquidity_universe not passed to BacktestPricing constructor"

        lookup = self._tvl_arrays.get(pair.internal_id)
        if lookup is None:
            raise LiquidityDataUnavailable(
                f"Could not read TVL/liquidity data for {pair} — "
                f"pair {pair.internal_id} not in pre-indexed liquidity data"
            )

        ts_seconds, values = lookup
        idx = self._backfill_index(ts_seconds, timestamp)
        if idx < 0:
            raise LiquidityDataUnavailable(
                f"Could not read TVL/liquidity data for {pair} — "
                f"no sample at or before {timestamp} within {self.data_delay_tolerance} tolerance"
            )
        return float(values[idx])

    #
    # Vault deposit/redemption availability (historical, backtest)
    #

    def _backfill_index(self, ts_seconds: np.ndarray, timestamp: AnyTimestamp) -> int:
        """Index of the most recent sample at or before ``timestamp`` within tolerance.

        Backward-fill via ``numpy.searchsorted`` on pre-indexed UNIX-second arrays. Returns
        ``-1`` when there is no sample at or before the timestamp, or the nearest one is older
        than :py:attr:`data_delay_tolerance`. Shared by TVL and vault-state lookups.
        """
        query_seconds = int(pd.Timestamp(timestamp).timestamp())
        idx = np.searchsorted(ts_seconds, query_seconds, side="right") - 1
        if idx < 0:
            return -1
        if query_seconds - int(ts_seconds[idx]) > int(self.data_delay_tolerance.total_seconds()):
            return -1
        return int(idx)

    def _lookup_vault_state(
        self,
        timestamp: AnyTimestamp | None,
        pair: TradingPairIdentifier | None,
    ) -> dict | None:
        """Return the in-tolerance vault availability sample at or before ``timestamp``.

        Returns ``None`` when there is no vault-state data, no pair/timestamp, no sample at or
        before the timestamp (pre-history), or the nearest sample is older than
        ``data_delay_tolerance``. Callers MUST treat ``None`` as "state unknown -> allowed",
        never as "closed".

        Uses the sample at or before the decision timestamp (``searchsorted`` right - 1), never a
        later same-bucket sample, so it introduces no look-ahead beyond the TVL/price candles.
        """
        if pair is None or timestamp is None:
            return None
        entry = self._vault_state_arrays.get(pair.internal_id)
        if entry is None:
            return None
        idx = self._backfill_index(entry["ts"], timestamp)
        if idx < 0:
            return None
        return {key: arr[idx] for key, arr in entry.items() if key != "ts"}

    @staticmethod
    def _state_cap(state: dict, key: str) -> Decimal | None:
        """Read a hard cap (``max_deposit`` / ``max_redeem``) as Decimal, or None if unknown."""
        cap = state.get(key)
        if cap is None:
            return None
        if isinstance(cap, float) and math.isnan(cap):
            return None
        return Decimal(str(cap))

    def _lookup_cap(self, ts, pair: TradingPairIdentifier, key: str) -> Decimal | None:
        state = self._lookup_vault_state(ts, pair)
        return None if state is None else self._state_cap(state, key)

    def get_max_deposit(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> Decimal | None:
        """Historical deposit hard cap for a vault at ``ts``, or None if unknown.

        Faithful data accessor, for parity with the live pricing models and for direct cap
        consumers / diagnostics. Backtest *rebalance gating* itself is open/closed only (via
        :py:meth:`can_deposit`); this value is reported, not enforced as a partial size limit.
        """
        return self._lookup_cap(ts, pair, "max_deposit")

    def get_max_redemption(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> Decimal | None:
        """Historical redemption hard cap for a vault at ``ts``, or None if unknown.

        See :py:meth:`get_max_deposit` for the report-vs-enforce distinction.
        """
        return self._lookup_cap(ts, pair, "max_redeem")

    def can_deposit(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> bool:
        override = self._vault_window_overrides.get(pair.internal_id) if pair is not None else None
        if override is not None:
            # Explicit backtest window override beats the (possibly stale) vault_state.
            return ts is None or override.is_deposit_open(ts)
        state = self._lookup_vault_state(ts, pair)
        if state is None:
            return True
        if state.get("deposits_open", -1) == 0:
            return False
        cap = self._state_cap(state, "max_deposit")
        if cap is not None and cap == 0:
            return False
        return True

    def check_redemption(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier | None,
        *,
        stage: RedemptionCheckStage = RedemptionCheckStage.unknown,
        position=None,
    ) -> RedemptionCheckResult:
        del position
        result = RedemptionCheckResult(
            timestamp=ts,
            stage=stage,
            can_redeem=True,
            pair_ticker=pair.get_ticker() if pair else None,
            vault_address=pair.pool_address if pair else None,
        )

        override = self._vault_window_overrides.get(pair.internal_id) if pair is not None else None
        if override is not None:
            # Explicit backtest window override beats the (possibly stale) vault_state.
            if ts is not None and not override.is_redemption_open(ts):
                result.can_redeem = False
                result.reason_code = RedemptionBlockReason.redemption_window_closed
                result.max_redemption = 0.0
                result.max_withdrawable = 0.0
                result.message = "Vault redemptions closed by backtest window override"
            return result

        state = self._lookup_vault_state(ts, pair)
        if state is None:
            return result

        cap = self._state_cap(state, "max_redeem")
        if state.get("redemption_open", -1) == 0 or (cap is not None and cap == 0):
            result.can_redeem = False
            result.reason_code = RedemptionBlockReason.vault_max_withdrawable_zero
            result.max_redemption = 0.0
            result.max_withdrawable = 0.0
            reason = state.get("redemption_closed_reason")
            result.message = reason if isinstance(reason, str) and reason else "Vault redemptions closed in historical data"
            return result

        # Report a positive historical cap for the Hyperliquid reduction path that consumes it.
        # The generic rebalance path enforces open/closed only, not partial size caps (out of
        # scope here; partial-cap sizing belongs in the size-risk model).
        if cap is not None:
            result.max_redemption = float(cap)
            result.max_withdrawable = float(cap)
        return result


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
        vault_state=universe.vault_state,
    )

