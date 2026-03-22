"""Test HypercoreVaultPricing returns candle-based share prices.

Validates the pricing model hack where :meth:`get_mid_price` returns
approximate share prices from candle data while execution pricing
(buy/sell) remains 1:1 USDC.

1. Build a mock candle universe with known share price data
2. Create a HypercoreVaultPricing with candle_universe attached
3. Verify get_mid_price returns candle share price, not 1.0
4. Verify get_buy_price/get_sell_price still return 1.0
5. Verify get_mid_price falls back to 1.0 when no candle data
"""

import datetime
from decimal import Decimal

import pandas as pd
import pytest
from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.cli.trade_ui_tui import _get_price
from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier


CHAIN_ID = 9999
PAIR_ID_WITH_DATA = 12345
PAIR_ID_WITHOUT_DATA = 99999
KNOWN_SHARE_PRICE = 1.42


@pytest.fixture()
def usdc_asset() -> AssetIdentifier:
    """USDC reserve asset."""
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address="0x000000000000000000000000000000000000dead",
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def vault_pair(usdc_asset: AssetIdentifier) -> TradingPairIdentifier:
    """A vault pair that has candle data."""
    base = AssetIdentifier(
        chain_id=CHAIN_ID,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="TEST-VAULT",
        decimals=6,
        internal_id=PAIR_ID_WITH_DATA,
    )
    return TradingPairIdentifier(
        base=base,
        quote=usdc_asset,
        pool_address="0x0000000000000000000000000000000000000002",
        exchange_address="0x0000000000000000000000000000000000000003",
        internal_id=PAIR_ID_WITH_DATA,
        internal_exchange_id=1,
        kind="vault",
    )


@pytest.fixture()
def vault_pair_no_data(usdc_asset: AssetIdentifier) -> TradingPairIdentifier:
    """A vault pair that has no candle data."""
    base = AssetIdentifier(
        chain_id=CHAIN_ID,
        address="0x0000000000000000000000000000000000000099",
        token_symbol="UNKNOWN-VAULT",
        decimals=6,
        internal_id=PAIR_ID_WITHOUT_DATA,
    )
    return TradingPairIdentifier(
        base=base,
        quote=usdc_asset,
        pool_address="0x0000000000000000000000000000000000000098",
        exchange_address="0x0000000000000000000000000000000000000003",
        internal_id=PAIR_ID_WITHOUT_DATA,
        internal_exchange_id=1,
        kind="vault",
    )


@pytest.fixture()
def candle_universe() -> GroupedCandleUniverse:
    """Build a GroupedCandleUniverse with one vault's share price data."""
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    timestamps = pd.date_range(end=now, periods=48, freq="h")

    rows = []
    for ts in timestamps:
        rows.append({
            "pair_id": PAIR_ID_WITH_DATA,
            "timestamp": ts,
            "open": KNOWN_SHARE_PRICE,
            "high": KNOWN_SHARE_PRICE,
            "low": KNOWN_SHARE_PRICE,
            "close": KNOWN_SHARE_PRICE,
            "volume": 0,
        })

    df = pd.DataFrame(rows)
    return GroupedCandleUniverse(df, time_bucket=TimeBucket.h1)


def test_mid_price_returns_candle_share_price(
    candle_universe: GroupedCandleUniverse,
    vault_pair: TradingPairIdentifier,
):
    """Verify get_mid_price reads approximate share price from candle data."""
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    ts = native_datetime_utc_now()
    mid_price = pricing.get_mid_price(ts, vault_pair)
    assert mid_price == pytest.approx(KNOWN_SHARE_PRICE, abs=0.01)


def test_buy_price_still_returns_one(
    candle_universe: GroupedCandleUniverse,
    vault_pair: TradingPairIdentifier,
):
    """Verify get_buy_price returns 1.0 for execution even with candle data."""
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    ts = native_datetime_utc_now()
    trade_pricing = pricing.get_buy_price(ts, vault_pair, Decimal("100"))
    assert trade_pricing.price == pytest.approx(1.0)
    assert trade_pricing.mid_price == pytest.approx(1.0)


def test_sell_price_still_returns_one(
    candle_universe: GroupedCandleUniverse,
    vault_pair: TradingPairIdentifier,
):
    """Verify get_sell_price returns 1.0 for execution even with candle data."""
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    ts = native_datetime_utc_now()
    trade_pricing = pricing.get_sell_price(ts, vault_pair, Decimal("50"))
    assert trade_pricing.price == pytest.approx(1.0)
    assert trade_pricing.mid_price == pytest.approx(1.0)


def test_mid_price_falls_back_without_candle_data(
    candle_universe: GroupedCandleUniverse,
    vault_pair_no_data: TradingPairIdentifier,
):
    """Verify get_mid_price returns 1.0 when no candle data for the pair."""
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    ts = native_datetime_utc_now()
    mid_price = pricing.get_mid_price(ts, vault_pair_no_data)
    assert mid_price == pytest.approx(1.0)


def test_mid_price_falls_back_without_candle_universe(
    vault_pair: TradingPairIdentifier,
):
    """Verify get_mid_price returns 1.0 when candle_universe is None."""
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=None,
    )
    ts = native_datetime_utc_now()
    mid_price = pricing.get_mid_price(ts, vault_pair)
    assert mid_price == pytest.approx(1.0)


def test_mid_price_uses_caller_timestamp(
    candle_universe: GroupedCandleUniverse,
    vault_pair: TradingPairIdentifier,
):
    """Verify get_mid_price respects the caller's timestamp, not wall-clock time.

    1. Create a pricing model backed by candle data (last 48 hours)
    2. Query at a timestamp far in the past (beyond 7-day tolerance)
    3. Assert it falls back to 1.0 because no candle is within tolerance
    4. Query at a timestamp within the candle range
    5. Assert it returns the known share price
    """
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )

    # 2-3. Timestamp far in the past — beyond 7-day tolerance window
    ancient_ts = datetime.datetime(2020, 1, 1)
    mid_price = pricing.get_mid_price(ancient_ts, vault_pair)
    assert mid_price == pytest.approx(1.0), (
        "get_mid_price should fall back to 1.0 for timestamps outside candle data range"
    )

    # 4-5. Timestamp within the candle range
    recent_ts = native_datetime_utc_now()
    mid_price = pricing.get_mid_price(recent_ts, vault_pair)
    assert mid_price == pytest.approx(KNOWN_SHARE_PRICE, abs=0.01)


def test_tui_get_price_displays_candle_share_price(
    candle_universe: GroupedCandleUniverse,
    vault_pair: TradingPairIdentifier,
):
    """Verify the TUI _get_price helper returns the candle share price.

    1. Create a pricing model backed by candle data
    2. Call the TUI's _get_price function
    3. Assert it returns a non-None value matching the known share price
    """
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    price = _get_price(pricing, vault_pair)
    assert price is not None, "TUI should display a price for a pair with candle data"
    assert price == pytest.approx(KNOWN_SHARE_PRICE, abs=0.01)


def test_tui_get_price_returns_none_without_data(
    candle_universe: GroupedCandleUniverse,
    vault_pair_no_data: TradingPairIdentifier,
):
    """Verify the TUI _get_price helper returns None when no candle data exists.

    1. Create a pricing model backed by candle data
    2. Call _get_price for a pair with no candles
    3. Assert it returns None (displayed as N/A in the TUI)
    """
    pricing = HypercoreVaultPricing(
        value_func=None,
        simulate=True,
        candle_universe=candle_universe,
    )
    price = _get_price(pricing, vault_pair_no_data)
    assert price is None, "TUI should show N/A for a pair without candle data"
