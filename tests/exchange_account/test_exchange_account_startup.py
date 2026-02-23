"""Test that exchange account positions survive the startup statistics pipeline.

When the trade executor starts, it runs update_position_valuations which
calls extract_long_short_stats_from_state. This triggers:

1. serialise_long_short_stats_as_json_table → trade analysis → func_check()
2. calculate_compounding_unrealised_trading_profitability → get_capital_tied_at_open_pct()

Both paths crash if portfolio_value_at_open is 0 or None, which happens
when exchange account positions are created on an empty portfolio.

These tests verify that:
- open_exchange_account_position sets portfolio_value_at_open correctly
- The statistics pipeline survives exchange account positions
- func_check handles None values in lists
"""

import datetime
from decimal import Decimal

import pytest

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.analysis.trade_analyser import build_trade_analysis, func_check
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
from tradeexecutor.visual.equity_curve import calculate_compounding_unrealised_trading_profitability


@pytest.fixture
def exchange_account_pair():
    """Create exchange account pair for testing."""
    usdc = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    derive_account = AssetIdentifier(
        chain_id=901,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": True,
        },
    ), usdc


def _create_state_with_exchange_account_position(pair, usdc) -> State:
    """Create a fresh state and open an exchange account position on it.

    Simulates what happens on the very first strategy cycle when
    the portfolio has no prior positions and no cash.
    """
    state = State()
    assert state.portfolio.calculate_total_equity() == 0, \
        "Sanity check: empty portfolio should have zero equity"

    open_exchange_account_position(
        state=state,
        strategy_cycle_at=datetime.datetime(2024, 1, 1),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=Decimal(1),
    )
    return state


def test_portfolio_value_at_open_set_on_empty_portfolio(exchange_account_pair):
    """portfolio_value_at_open must be non-zero even on an empty portfolio.

    Without this, get_capital_tied_at_open_pct() raises LegacyDataException
    and the startup statistics pipeline crashes.
    """
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    position = list(state.portfolio.open_positions.values())[0]
    assert position.portfolio_value_at_open is not None
    assert position.portfolio_value_at_open > 0, \
        f"portfolio_value_at_open should be positive, got {position.portfolio_value_at_open}"


def test_capital_tied_at_open_pct_does_not_crash(exchange_account_pair):
    """get_capital_tied_at_open_pct must not raise for exchange account positions."""
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    position = list(state.portfolio.open_positions.values())[0]
    pct = position.get_capital_tied_at_open_pct()
    assert pct > 0


def test_size_relative_profit_does_not_crash(exchange_account_pair):
    """get_size_relative_unrealised_or_realised_profit_percent must not raise.

    This is called by calculate_compounding_unrealised_trading_profitability
    during the startup statistics pipeline.
    """
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    position = list(state.portfolio.open_positions.values())[0]
    # Should not raise LegacyDataException
    pct = position.get_size_relative_unrealised_or_realised_profit_percent()
    assert isinstance(pct, (int, float))


def test_compounding_unrealised_profitability_does_not_crash(exchange_account_pair):
    """calculate_compounding_unrealised_trading_profitability must survive exchange account positions.

    This is called during update_position_valuations at startup.
    """
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    # Should not raise
    result = calculate_compounding_unrealised_trading_profitability(state)
    assert result is not None
    assert len(result) == 1


def test_trade_analysis_summary_does_not_crash(exchange_account_pair):
    """build_trade_analysis + calculate_all_summary_stats_by_side must survive.

    This is the other crash path: the trade analyser iterates positions
    and calls func_check(loss_risk_at_open_pc, max) which can fail
    if the list contains None values.
    """
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    analysis = build_trade_analysis(state.portfolio)
    # Should not raise
    summary = analysis.calculate_all_summary_stats_by_side(state=state)
    assert summary is not None


def test_serialise_long_short_stats_does_not_crash(exchange_account_pair):
    """serialise_long_short_stats_as_json_table must not crash at startup.

    This is the top-level function called from extract_long_short_stats_from_state
    during run_live startup. It exercises both crash paths:
    1. trade analysis summary statistics
    2. compounding unrealised profitability
    """
    pair, usdc = exchange_account_pair
    state = _create_state_with_exchange_account_position(pair, usdc)

    # Should not raise
    result = serialise_long_short_stats_as_json_table(live_state=state)
    assert "live_stats" in result
    assert result["live_stats"] is not None


def test_func_check_filters_none_values():
    """func_check must handle lists containing None values.

    When portfolio_value_at_open is not set, get_capital_tied_at_open
    returns None, which gets appended to loss_risk_at_open_pc.
    max([None, None]) raises TypeError: '>' not supported between NoneType.
    """
    # List with all None values — should return None, not crash
    assert func_check([None, None], max) is None
    assert func_check([None, None], min) is None

    # Mixed None and real values — should ignore None
    assert func_check([None, 5.0, None, 3.0], max) == 5.0
    assert func_check([None, 5.0, None, 3.0], min) == 3.0

    # Empty list — should return None
    assert func_check([], max) is None

    # Normal list without None — should work as before
    assert func_check([1.0, 2.0, 3.0], max) == 3.0
