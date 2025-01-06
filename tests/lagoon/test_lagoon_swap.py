"""Test Lagoon vault trades."""

import datetime
import os
from decimal import Decimal

import pytest

from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.lagoon.vault import LagoonVault
from eth_defi.trace import assert_transaction_success_with_explanation
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


@pytest.fixture()
def deposited_vault(
    web3,
    depositor,
    base_usdc_token,
    vault_strategy_universe,
    asset_manager,
    automated_lagoon_vault,
) -> tuple[LagoonVault, State]:
    """Vault with deposited cash.

    - Sync and deposit $399 trading balance so we can perform swap tests
    """

    vault = automated_lagoon_vault.vault
    usdc = base_usdc_token
    strategy_universe = vault_strategy_universe
    state = State()
    portfolio = state.portfolio
    usdc_asset = strategy_universe.get_reserve_asset()

    sync_model = LagoonVaultSyncModel(
        vault=vault,
        hot_wallet=asset_manager,
    )

    sync_model.sync_initial(
        state,
        reserve_asset=usdc_asset,
        reserve_token_price=1.0,
    )

    # Do initial deposit
    # Deposit 399.00 USDC into the vault from the first user
    usdc_amount = Decimal(399.00)
    raw_usdc_amount = usdc.convert_to_raw(usdc_amount)
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, raw_usdc_amount)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)
    assert len(events) == 1
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(399)

    return vault, state


def test_lagoon_swap_uniswap_v2(
    deposited_vault: LagoonAutomatedDeployment,
    vault_strategy_universe: TradingStrategyUniverse,
    asset_manager: HotWallet,
    lagoon_execution_model: LagoonExecution,
    lagoon_pricing_model: GenericPricing,
    lagoon_routing_model:RoutingModel,
):
    """Do a deposit to Lagoon vault and perform Uniswap v2 token buy (three legs)."""

    vault, state = deposited_vault
    strategy_universe = vault_strategy_universe
    execution_model = lagoon_execution_model
    routing_model = lagoon_routing_model

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=lagoon_pricing_model,
        default_slippage_tolerance=0.20,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v2", "KEYCAT", "WETH"))
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
