"""Test that exchange account positions survive the startup statistics pipeline.

When the trade executor starts, it runs update_position_valuations which
calls extract_long_short_stats_from_state. This triggers:

1. serialise_long_short_stats_as_json_table → trade analysis → func_check()
2. calculate_compounding_unrealised_trading_profitability → get_capital_tied_at_open_pct()

Both paths crash if portfolio_value_at_open is 0 or None, which happens
when exchange account positions are created on an empty portfolio.
"""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.analysis.trade_analyser import func_check
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table


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


def test_startup_statistics_with_exchange_account_position(exchange_account_pair):
    """The full startup statistics pipeline must not crash with exchange account positions.

    Simulates what happens on the very first strategy cycle: an exchange account
    position is created on an empty portfolio (zero equity), then the startup
    revaluation calls serialise_long_short_stats_as_json_table. This exercises:

    - portfolio_value_at_open is set correctly (non-zero) by open_exchange_account_position
    - get_capital_tied_at_open_pct() does not raise LegacyDataException
    - get_size_relative_unrealised_or_realised_profit_percent() survives
    - calculate_compounding_unrealised_trading_profitability() survives
    - build_trade_analysis + calculate_all_summary_stats_by_side() survives
    - func_check(loss_risk_at_open_pc, max) does not crash on None values
    """
    pair, usdc = exchange_account_pair

    state = State()
    assert state.portfolio.calculate_total_equity() == 0

    open_exchange_account_position(
        state=state,
        strategy_cycle_at=datetime.datetime(2024, 1, 1),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=Decimal(1),
    )

    position = list(state.portfolio.open_positions.values())[0]
    assert position.portfolio_value_at_open > 0
    assert position.get_capital_tied_at_open_pct() > 0

    # This is the top-level function called from extract_long_short_stats_from_state
    # during run_live startup — exercises both crash paths end-to-end
    result = serialise_long_short_stats_as_json_table(live_state=state)
    assert "live_stats" in result
    assert result["live_stats"] is not None


def test_func_check_filters_none_values():
    """func_check must handle lists containing None values.

    When portfolio_value_at_open is not set, get_capital_tied_at_open
    returns None, which gets appended to loss_risk_at_open_pc.
    max([None, None]) raises TypeError: '>' not supported between NoneType.
    """
    assert func_check([None, None], max) is None
    assert func_check([None, None], min) is None
    assert func_check([None, 5.0, None, 3.0], max) == 5.0
    assert func_check([None, 5.0, None, 3.0], min) == 3.0
    assert func_check([], max) is None
    assert func_check([1.0, 2.0, 3.0], max) == 3.0
