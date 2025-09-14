"""Multi-staged deposit and redemption tests for ERC-7540 and Gains vaults."""


import datetime
from decimal import Decimal

import pytest

from eth_defi.ipor.vault import IPORVault
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager



def test_trade_as_lagoon(
    vault: IPORVault,
    strategy_universe,
    execution_model,
    routing_model: GenericRouting,
    pricing_model,
    sync_model: HotWalletSyncModel,
    base_usdc: AssetIdentifier,
):
    """Do a deposit to Lagoon vault and perform Uniswap v2 token buy (three legs)."""

    state = State()
    pair = strategy_universe.get_pair_by_smart_contract(vault.address)

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

    # Deposit to the vault
    trades = position_manager.open_spot(
        pair,
        value=10.00,
    )
    t = trades[0]

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert t.is_success(), f"Trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert t.is_buy()
    assert t.executed_price == pytest.approx(1.0335669715951414)
    assert t.executed_quantity == pytest.approx(Decimal(9.67523177))
    assert t.executed_reserve == 10

    position = position_manager.get_current_position_for_pair(pair)
    assert position.get_quantity() == pytest.approx(Decimal(9.67523177))

    # Then redeem shares back
    trades = position_manager.close_all()
    assert len(trades) == 1
    t = trades[0]

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert t.is_success(), f"Trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert t.is_sell()
    assert t.planned_quantity == pytest.approx(Decimal(-9.67523177))
    assert t.executed_price == pytest.approx(1.0335668671701836)
    assert t.executed_quantity == pytest.approx(Decimal(-9.67523178))
    assert t.executed_reserve == pytest.approx(Decimal('9.999999'))

    assert position.is_closed()
