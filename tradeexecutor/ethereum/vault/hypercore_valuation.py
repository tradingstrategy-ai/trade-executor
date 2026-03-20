"""Valuation and pricing models for Hypercore native vault positions.

Hypercore vault equity is queried via the Hyperliquid info API,
not from on-chain contracts.

Position model:

- **quantity** tracks cumulative USDC deposited minus withdrawn.
- **price** = equity / quantity (the return multiplier per deposited USDC).
- **value** = quantity × price = equity (by construction).

Trade pricing is always 1:1 USDC — equity growth is captured solely
by the valuation model, not the pricing model.

In simulate mode (Anvil forks), the Hyperliquid API has no data for the
forked Safe address, so the API is skipped and a 1:1 USDC price is assumed.
"""

import datetime
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from tradingstrategy.candle import GroupedCandleUniverse
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    HyperliquidSession,
    create_hyperliquid_session,
)
from eth_defi.hyperliquid.vault import HyperliquidVault, VaultInfo

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.testing.hypercore_replay import (
    HypercoreReplaySnapshot,
    HypercoreVaultMarketDataSource,
)

logger = logging.getLogger(__name__)


# Must match eth_defi.hyperliquid.vault_data_export.LEADER_FRACTION_WARNING_THRESHOLD.
LEADER_FRACTION_WARNING_THRESHOLD = 0.055


def get_hypercore_deposit_closed_reason(info: VaultInfo) -> str | None:
    """Return a descriptive reason why Hypercore deposits are not allowed."""
    return _get_hypercore_deposit_closed_reason_from_flags(
        is_closed=info.is_closed,
        allow_deposits=info.allow_deposits,
        relationship_type=info.relationship_type,
        leader_fraction=info.leader_fraction,
    )


def _get_hypercore_deposit_closed_reason_from_flags(
    *,
    is_closed: bool,
    allow_deposits: bool,
    relationship_type: str,
    leader_fraction: float | Decimal | None,
) -> str | None:
    """Return a descriptive reason why Hypercore deposits are not allowed."""
    if is_closed:
        return "Vault is permanently closed"

    # HLP parent deposits remain open even if the API reports allowDeposits=False.
    if relationship_type == "parent":
        return None

    if not allow_deposits:
        return "Vault deposits disabled by leader"

    if leader_fraction is not None and float(leader_fraction) < LEADER_FRACTION_WARNING_THRESHOLD:
        return "Leader share of the vault capital near allowed Hyperliquid minimum and new capital may not be accepted"

    return None


class HypercoreVaultPricing(PricingModel):
    """Pricing model for Hypercore vault deposit/withdrawal trades.

    .. warning::

        This pricing model contains several deliberate hacks because
        Hyperliquid has no on-chain concept of vault share price, unlike
        ERC-4626 vaults where share price is computable from on-chain
        contract state.

    Design compromises and hacks
    ----------------------------

    **1:1 USDC execution pricing (hack)**

    :meth:`get_buy_price` and :meth:`get_sell_price` always return 1.0 USDC
    per unit. This is because Hypercore vault deposits and withdrawals
    exchange USDC at face value — there is no slippage or price impact on
    the deposit/withdrawal itself. Equity growth (profit/loss from the
    vault's trading activity) is tracked entirely by the valuation model
    (:class:`HypercoreVaultValuator`), not by trade pricing. This means
    the pricing model deliberately lies about the asset's market value to
    make the deposit/withdrawal accounting work correctly.

    **Approximate share prices from data pipeline (hack)**

    :meth:`get_mid_price` returns an *approximate* share price read from
    the Trading Strategy data pipeline's cleaned vault price bundle
    (``~/.tradingstrategy/vaults/cleaned-vault-prices-1h.parquet``).
    These are **not** real-time on-chain prices — they are periodic
    snapshots collected by the data pipeline and may lag by hours or
    days depending on the refresh frequency. They are suitable for
    display purposes (TUI, dashboards) but must not be used for trade
    execution decisions. When no candle data is available, falls back
    to 1.0.

    **Post-deposit valuation via Hypercore API**

    After a deposit, the position's actual value is determined by
    querying the Hyperliquid info API through :class:`HypercoreVaultValuator`,
    which computes ``price = equity / quantity``. This is the only
    source of truth for position value in live trading.

    :param value_func:
        Callable that fetches current vault equity for a pair via
        the Hyperliquid API. Used in live trading only.

    :param safe_address_resolver:
        Resolves the Safe address that holds vault deposits for a pair.

    :param session_factory:
        Creates Hyperliquid API sessions for vault info queries.

    :param simulate:
        When ``True``, skip the Hyperliquid API and use 1.0 USDC per unit.
        Used in Anvil fork mode where the API has no data for the forked Safe.

    :param market_data_source:
        Replay data source for backtesting and tests. Provides vault
        snapshots (TVL, gating flags, equity) without API calls.

    :param candle_universe:
        Historical vault share prices from the Trading Strategy data
        pipeline, used by :meth:`get_mid_price` for display purposes
        only (TUI pair tables, dashboards). These are approximations
        derived from periodic vault snapshots — not live prices.
        The data may lag by hours or days. When ``None``,
        :meth:`get_mid_price` falls back to 1.0.
    """

    def __init__(
        self,
        value_func: Callable[[TradingPairIdentifier], Decimal] | None,
        safe_address_resolver: Callable[[TradingPairIdentifier], str | None] | None = None,
        session_factory: Callable[[TradingPairIdentifier], HyperliquidSession] | None = None,
        simulate: bool = False,
        market_data_source: HypercoreVaultMarketDataSource | None = None,
        candle_universe: "GroupedCandleUniverse | None" = None,
    ):
        self.value_func = value_func
        self.safe_address_resolver = safe_address_resolver
        self.session_factory = session_factory
        self.simulate = simulate
        self.market_data_source = market_data_source
        self.candle_universe = candle_universe
        self._session_cache: dict[str, HyperliquidSession] = {}

    def _make_pricing(self, pair: TradingPairIdentifier, token_in: Decimal | None = None, token_out: Decimal | None = None) -> TradePricing:
        """Build a :class:`TradePricing` for vault deposits/withdrawals.

        Vault deposits and withdrawals are 1:1 USDC, so the trade price
        is always 1.0.  The vault equity growth is reflected in the
        valuation model (per-unit price = equity / quantity), not here.
        """
        price = 1.0
        return TradePricing(
            price=price,
            mid_price=price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=native_datetime_utc_now(),
            block_number=None,
            token_in=token_in,
            token_out=token_out,
        )

    def get_sell_price(self, ts, pair, quantity) -> TradePricing:
        return self._make_pricing(pair, token_in=quantity)

    def get_buy_price(self, ts, pair, reserve) -> TradePricing:
        return self._make_pricing(pair, token_in=reserve)

    def _get_share_price_from_candles(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> float | None:
        """Look up the vault share price from candle data at a given timestamp.

        Returns the approximate share price from the Trading Strategy
        data pipeline, or ``None`` if no candle data is available.
        These are periodic snapshots, not real-time prices.

        :param ts:
            Timestamp to look up the price at. Uses a 7-day tolerance
            to find the nearest candle.
        """
        if self.candle_universe is None:
            return None
        try:
            when = pd.Timestamp(ts, tz="UTC")
            price, _ = self.candle_universe.get_price_with_tolerance(
                pair.internal_id,
                when=when,
                tolerance=pd.Timedelta("7D"),
            )
            return float(price)
        except Exception:
            return None

    def get_mid_price(self, ts, pair) -> float:
        # Return approximate share price from candle data if available.
        # This is a data-pipeline approximation, not a live price.
        # Falls back to 1.0 (the deposit/withdrawal face value).
        candle_price = self._get_share_price_from_candles(ts, pair)
        if candle_price is not None:
            return candle_price
        return 1.0

    def get_pair_fee(self, ts, pair):
        return 0.0

    def get_pair_for_id(self, internal_id):
        raise NotImplementedError("Hypercore vault pricing does not support pair lookup by ID")

    def _get_safe_address(self, pair: TradingPairIdentifier) -> str | None:
        if self.safe_address_resolver is None:
            return None
        return self.safe_address_resolver(pair)

    def _get_session(self, pair: TradingPairIdentifier) -> HyperliquidSession:
        if self.session_factory is not None:
            return self.session_factory(pair)

        api_url = HYPERLIQUID_TESTNET_API_URL if pair.other_data.get("exchange_is_testnet", False) else HYPERLIQUID_API_URL
        session = self._session_cache.get(api_url)
        if session is None:
            session = create_hyperliquid_session(api_url=api_url)
            self._session_cache[api_url] = session
        return session

    def _get_vault_info(self, pair: TradingPairIdentifier) -> VaultInfo:
        vault_address = pair.pool_address
        assert vault_address, f"No pool_address set for Hypercore vault pair: {pair}"
        session = self._get_session(pair)
        return HyperliquidVault(session=session, vault_address=vault_address).fetch_info()

    def _get_market_snapshot(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
        *,
        net_deposited_usdc: Decimal | None = None,
        current_equity_usd: Decimal | None = None,
    ) -> HypercoreReplaySnapshot:
        assert self.market_data_source is not None
        safe_address = self._get_safe_address(pair)
        return self.market_data_source.get_snapshot(
            ts or native_datetime_utc_now(),
            pair,
            safe_address=safe_address,
            net_deposited_usdc=net_deposited_usdc,
            current_equity_usd=current_equity_usd,
        )

    def get_usd_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> float:
        if self.market_data_source is not None:
            snapshot = self._get_market_snapshot(timestamp, pair)
            return float(snapshot.tvl_usd)

        raise NotImplementedError("Hypercore vault TVL needs a replay market-data source in live-style tests")

    def get_max_deposit(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> Decimal | None:
        if self.market_data_source is not None:
            snapshot = self._get_market_snapshot(ts, pair)
            reason = _get_hypercore_deposit_closed_reason_from_flags(
                is_closed=snapshot.is_closed,
                allow_deposits=snapshot.allow_deposits,
                relationship_type=snapshot.relationship_type,
                leader_fraction=snapshot.leader_fraction,
            )
            if reason is not None:
                logger.info("Hypercore replay vault %s deposits closed: %s", pair, reason)
                return Decimal(0)
            return None

        if self.simulate:
            return None

        info = self._get_vault_info(pair)
        reason = get_hypercore_deposit_closed_reason(info)
        if reason is not None:
            logger.info("Hypercore vault %s deposits closed: %s", pair, reason)
            return Decimal(0)
        return None

    def get_max_redemption(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> Decimal | None:
        if self.market_data_source is not None:
            current_equity_usd = None
            if self.value_func is not None and not self.simulate:
                current_equity_usd = self.value_func(pair)

            snapshot = self._get_market_snapshot(
                ts,
                pair,
                current_equity_usd=current_equity_usd,
            )

            if not snapshot.lockup_expired:
                return Decimal(0)

            if snapshot.max_withdrawable_usd is None:
                return None

            return snapshot.max_withdrawable_usd

        if self.simulate:
            return None

        safe_address = self._get_safe_address(pair)
        if safe_address is None:
            logger.warning("Cannot resolve safe address for Hypercore redemption check: %s", pair)
            return None

        info = self._get_vault_info(pair)
        if info.max_withdrawable <= 0:
            return Decimal(0)

        vault_address = pair.pool_address
        assert vault_address, f"No pool_address set for Hypercore vault pair: {pair}"
        eq = fetch_user_vault_equity(
            self._get_session(pair),
            user=safe_address,
            vault_address=vault_address,
            bypass_cache=True,
        )
        if eq is None:
            return Decimal(0)

        if not eq.is_lockup_expired:
            return Decimal(0)

        return min(eq.equity, info.max_withdrawable)

    def can_deposit(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> bool:
        if self.market_data_source is not None:
            snapshot = self._get_market_snapshot(ts, pair)
            return _get_hypercore_deposit_closed_reason_from_flags(
                is_closed=snapshot.is_closed,
                allow_deposits=snapshot.allow_deposits,
                relationship_type=snapshot.relationship_type,
                leader_fraction=snapshot.leader_fraction,
            ) is None

        if self.simulate:
            return True

        info = self._get_vault_info(pair)
        return get_hypercore_deposit_closed_reason(info) is None

    def can_redeem(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
    ) -> bool:
        if self.market_data_source is not None:
            snapshot = self._get_market_snapshot(ts, pair)
            if not snapshot.lockup_expired:
                return False

            if snapshot.max_withdrawable_usd is None:
                return True

            return snapshot.max_withdrawable_usd > 0

        if self.simulate:
            return True

        max_redemption = self.get_max_redemption(ts, pair)
        if max_redemption is None:
            return True
        return max_redemption > 0


class HypercoreVaultValuator(ValuationModel):
    """Re-value Hypercore vault positions using the Hyperliquid info API.

    Queries the vault equity for the Safe address and computes
    a per-unit price so that ``value = quantity × price = equity``.

    - **quantity** = cumulative USDC deposited minus withdrawn
    - **price** = equity / quantity (return multiplier per deposited USDC)

    :param simulate:
        When ``True``, skip the Hyperliquid API and use 1.0 USDC per unit.
        Used in Anvil fork mode where the API has no data for the forked Safe.
    """

    def __init__(
        self,
        value_func: Callable[[TradingPairIdentifier], Decimal] | None,
        simulate: bool = False,
        market_data_source: HypercoreVaultMarketDataSource | None = None,
    ):
        self.value_func = value_func
        self.simulate = simulate
        self.market_data_source = market_data_source

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        assert position.is_vault(), f"Not a vault position: {position}"

        position.last_pricing_at = ts

        if self.market_data_source is not None:
            snapshot = self.market_data_source.get_snapshot(
                ts,
                position.pair,
                net_deposited_usdc=position.get_quantity(),
            )
            equity = snapshot.equity_usd
            assert equity is not None, f"Replay snapshot did not produce equity for {position}"
        elif self.simulate:
            # Anvil fork: Hyperliquid API has no data for forked Safe
            equity = position.get_quantity()
            logger.info(
                "Hypercore vault position %s: simulate mode, using quantity %s as equity",
                position, equity,
            )
        else:
            assert self.value_func is not None, "value_func missing for live Hypercore valuation"
            try:
                equity = self.value_func(position.pair)
            except Exception as e:
                # Intentionally crash the tick cycle: if the API is down we
                # cannot value the position and must halt rather than use
                # stale data.
                logger.error(
                    "Failed to get Hypercore vault equity for position %s: %s",
                    position, e,
                )
                raise

        old_price = position.last_token_price
        old_value = position.get_value()

        # Compute per-unit price so that value = quantity * price = equity.
        # Position quantity tracks cumulative USDC deposited/withdrawn;
        # the price reflects the return on each deposited USDC.
        quantity = position.get_quantity()
        if float(quantity) > 0:
            new_price = float(equity) / float(quantity)
        else:
            new_price = 1.0

        new_value = position.revalue_base_asset(ts, new_price)

        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=position.get_quantity(),
        )

        position.last_token_price = new_price

        logger.info(
            "Hypercore vault position %s, valuation updated: equity=$%.2f, old=$%.2f, new=$%.2f",
            position, equity, old_value, new_value,
        )

        return evt
