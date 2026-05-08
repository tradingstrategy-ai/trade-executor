"""Test sequential trade execution for CCTP bridge-dependent batches.

Verifies that BacktestExecution detects CCTP bridge trades and switches
to per-trade start/simulate/settle ordering, so that bridge positions
exist before satellite trades try to allocate from them.
"""
import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType


#: Arbitrum native USDC address
USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"

#: Base native USDC address
USDC_BASE_ADDRESS = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

#: Dummy WETH on Base
WETH_BASE_ADDRESS = "0x4200000000000000000000000000000000000006"


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum — the reserve currency."""
    return AssetIdentifier(
        chain_id=42161,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base — destination chain stablecoin."""
    return AssetIdentifier(
        chain_id=8453,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def weth_base() -> AssetIdentifier:
    """WETH on Base — a satellite chain token."""
    return AssetIdentifier(
        chain_id=8453,
        address=WETH_BASE_ADDRESS,
        token_symbol="WETH",
        decimals=18,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        exchange_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def spot_pair(weth_base: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """A simple satellite-chain spot pair (WETH/USDC on Base)."""
    return TradingPairIdentifier(
        base=weth_base,
        quote=usdc_base,
        pool_address="0xd0b53d9277642d899df5c87a3966a349a798f224",
        exchange_address="0x33128a8fc17869897dce68ed026d694621f6fdfd",
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
    )


@pytest.fixture()
def wallet(usdc_arbitrum: AssetIdentifier) -> SimulatedWallet:
    """A simulated wallet seeded with 10 000 USDC on Arbitrum."""
    w = SimulatedWallet()
    w.set_balance(usdc_arbitrum, Decimal(10_000))
    return w


@pytest.fixture()
def execution(wallet: SimulatedWallet) -> BacktestExecution:
    """BacktestExecution wired to the simulated wallet."""
    return BacktestExecution(wallet=wallet)


@pytest.fixture()
def state_with_reserves(usdc_arbitrum: AssetIdentifier) -> State:
    """A fresh State with 10 000 USDC reserves on Arbitrum."""
    state = State()
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    return state


def test_backtest_bridge_then_spot_sequential(
    state_with_reserves: State,
    execution: BacktestExecution,
    wallet: SimulatedWallet,
    cctp_pair: TradingPairIdentifier,
    spot_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify that a bridge-out buy followed by a satellite spot buy
    executes correctly via the sequential path.

    1. Create two trades: a CCTP bridge-out buy and a satellite spot buy
    2. Sort them by execution sort position (bridge first)
    3. Call execute_trades() — should detect bridge trades and use sequential path
    4. Verify the bridge trade completes and a bridge position exists
    5. Verify the satellite spot trade completes using bridge capital
    6. Verify wallet balances are consistent
    """
    ts = datetime.datetime(2025, 6, 1)
    state = state_with_reserves

    # 1. Create a bridge-out buy: 1000 USDC from Arbitrum to Base
    _, bridge_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(1000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 2. Create a satellite spot buy: 500 USDC worth of WETH on Base
    _, spot_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=spot_pair,
        quantity=None,
        reserve=Decimal(500),
        assumed_price=2000.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Sort by execution sort position (bridge-out comes before spot buy)
    trades = sorted(
        [bridge_trade, spot_trade],
        key=lambda t: t.get_execution_sort_position(),
    )

    # Set up a minimal routing model/state for the backtest
    routing_model = BacktestRoutingIgnoredModel(
        reserve_token_address=usdc_arbitrum.address,
    )
    routing_state = BacktestRoutingState(pair_universe=None, wallet=wallet)

    # 4. Execute — should take the sequential path because of the bridge trade
    execution.execute_trades(
        ts=ts,
        state=state,
        trades=trades,
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )

    # 5. Verify bridge trade completed
    assert bridge_trade.is_success()
    assert bridge_trade.executed_quantity == Decimal(1000)

    # 6. Verify bridge position exists with correct quantity
    bridge_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.is_cctp_bridge()
    ]
    assert len(bridge_positions) == 1
    bridge_pos = bridge_positions[0]
    assert bridge_pos.get_quantity() == Decimal(1000)

    # 7. Verify spot trade completed
    assert spot_trade.is_success()
    assert spot_trade.executed_quantity == pytest.approx(Decimal(500) / Decimal(2000), abs=Decimal("0.001"))

    # 8. Verify reserves decreased by the bridge amount only
    # (the spot trade draws from bridge capital, not from the main reserves)
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity == Decimal(10_000) - Decimal(1000)

    # 9. Verify bridge position has capital allocated to the satellite trade
    assert bridge_pos.bridge_capital_allocated == Decimal(500)


def test_backtest_no_bridge_uses_batch_path(
    state_with_reserves: State,
    execution: BacktestExecution,
    wallet: SimulatedWallet,
    spot_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
):
    """Verify that a backtest without bridge trades still uses the fast
    batch path (start_execution_all) with no regression.

    1. Create a single spot buy trade (no bridge trades)
    2. Call execute_trades() with a patched start_execution_all
    3. Assert start_execution_all was called (batch path)
    4. Verify the trade succeeds
    """
    ts = datetime.datetime(2025, 6, 1)
    state = state_with_reserves

    # The spot pair uses Base USDC as quote — but the wallet/reserves hold
    # Arbitrum USDC. For a pure non-bridge test we need the wallet to have
    # the quote token so simulate_spot doesn't fail.
    wallet.set_balance(usdc_base, Decimal(10_000))

    # 1. Create a spot buy on Base: 500 USDC worth of WETH
    _, spot_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=spot_pair,
        quantity=None,
        reserve=Decimal(500),
        assumed_price=2000.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    trades = [spot_trade]

    routing_model = BacktestRoutingIgnoredModel(
        reserve_token_address=usdc_arbitrum.address,
    )
    routing_state = BacktestRoutingState(pair_universe=None, wallet=wallet)

    # 2. Patch start_execution_all to verify it is called
    with patch.object(State, "start_execution_all", wraps=state.start_execution_all) as mock_start_all:
        execution.execute_trades(
            ts=ts,
            state=state,
            trades=trades,
            routing_model=routing_model,
            routing_state=routing_state,
            check_balances=False,
        )

        # 3. Assert batch path was used
        mock_start_all.assert_called_once()

    # 4. Verify the trade succeeded
    assert spot_trade.is_success()
