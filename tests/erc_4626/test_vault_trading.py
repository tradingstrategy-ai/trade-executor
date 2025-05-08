"""Test Lagoon vault trades."""

import datetime
import os
from decimal import Decimal

import pytest
from web3 import Web3

from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.hotwallet import HotWallet
from eth_defi.ipor.vault import IPORVault
from eth_defi.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.lagoon.vault import LagoonVault
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.lagoon.execution import LagoonExecution

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.vault.vault_routing import VaultRouting
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


def test_vault_routing(
    vault: IPORVault,
    routing_model: GenericRouting,
    strategy_universe: TradingStrategyUniverse,
):
    """Check we know how to route vault trades."""
    pair = strategy_universe.get_pair_by_smart_contract(vault.address)
    assert pair.is_vault()
    assert pair.get_vault_features() == {ERC4626Feature.ipor_like}
    assert pair.get_vault_protocol() == "ipor"

    routing_id = routing_model.pair_configurator.match_router(pair)
    protocol_config = routing_model.pair_configurator.get_config(routing_id)
    assert protocol_config.routing_id.router_name == "vault"
    assert protocol_config.routing_id.exchange_slug is None
    assert isinstance(protocol_config.routing_model, VaultRouting)


def test_vault_trading_deposit(
    vault: IPORVault,
    strategy_universe,
    execution_model,
    routing_model: GenericRouting,
    pricing_model,
):
    """Do a deposit to Lagoon vault and perform Uniswap v2 token buy (three legs)."""

    state = State()

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

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
    assert 0 < t.executed_price < 1
    assert t.executed_quantity > 1000  # Keycat tokens
    assert t.executed_reserve > 0

    # Then sell Keycat
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
    assert 0 < t.executed_price < 1
    assert t.executed_quantity < 1000  # DogInMe tokens
    assert t.executed_reserve > 0
