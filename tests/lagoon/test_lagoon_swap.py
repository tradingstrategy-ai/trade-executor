"""Test Lagoon vault trades."""

import datetime
import os
from decimal import Decimal

import pytest
from web3 import Web3

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
def sync_model(
    automated_lagoon_vault,
    asset_manager,
) -> LagoonVaultSyncModel:
    sync_model = LagoonVaultSyncModel(
        vault=automated_lagoon_vault.vault,
        hot_wallet=asset_manager,
        unit_testing=True,
    )
    return sync_model


@pytest.fixture()
def deposited_vault(
    web3,
    depositor,
    base_usdc,
    base_usdc_token,
    vault_strategy_universe,
    asset_manager,
    automated_lagoon_vault,
    sync_model,
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
    events = sync_model.sync_treasury(
        cycle,
        state,
        supported_reserves=[base_usdc],
        post_valuation=True,
    )
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


def test_lagoon_swap_uniswap_v3(
    deposited_vault: LagoonAutomatedDeployment,
    vault_strategy_universe: TradingStrategyUniverse,
    asset_manager: HotWallet,
    lagoon_execution_model: LagoonExecution,
    lagoon_pricing_model: GenericPricing,
    lagoon_routing_model:RoutingModel,
):
    """Do a deposit to Lagoon vault and perform Uniswap v3 token buy/sell (three legs)."""

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

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v3", "DogInMe", "WETH"))
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

    # Then sell DogInMe
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


def test_lagoon_redemption_queue(
    web3: Web3,
    deposited_vault: LagoonAutomatedDeployment,
    vault_strategy_universe: TradingStrategyUniverse,
    asset_manager: HotWallet,
    lagoon_execution_model: LagoonExecution,
    lagoon_pricing_model: GenericPricing,
    lagoon_routing_model:RoutingModel,
    depositor,
    sync_model,
):
    """Redemption queue valeu is correctly filled in when we cannot satisfy all redemptinos instantly."""

    vault, state = deposited_vault
    strategy_universe = vault_strategy_universe
    execution_model = lagoon_execution_model
    routing_model = lagoon_routing_model

    #
    # Start by depositing, creating an open position in DogInMe token
    #

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=lagoon_pricing_model,
        default_slippage_tolerance=0.20,
    )

    assert position_manager.get_current_cash() == 399

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v3", "DogInMe", "WETH"))
    trades = position_manager.open_spot(
        pair,
        value=300.00,
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

    assert position_manager.get_current_cash() == 99

    #
    # Start a redemption
    #

    redeem_amount = 300

    # Make sure shares are in the wallet before redemption
    bound_func = vault.finalise_deposit(depositor)
    tx_hash = bound_func.transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Try to redeem 8 USDC - we do not have enough cash
    shares_to_redeem_raw = vault.share_token.convert_to_raw(Decimal(redeem_amount))
    bound_func = vault.request_redeem(depositor, shares_to_redeem_raw)
    tx_hash = bound_func.transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Settle to see the redemption,
    # was not instantly settled because it is more than we have cash
    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        post_valuation=True,
    )

    assert position_manager.get_pending_redemptions() == pytest.approx(300)
    assert position_manager.get_current_cash() == 99

    # Then sell DogInMe
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

    # We do get cash now, but it is not yet settled
    assert position_manager.get_pending_redemptions() == pytest.approx(300)
    assert position_manager.get_current_cash() > 300

    # Settle to clear the redemption,
    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        post_valuation=True,
    )

    # Now redemptions are cleared,
    # the redeem balance is on vault smart contract
    assert position_manager.get_pending_redemptions() == pytest.approx(0)
    assert 0 < position_manager.get_current_cash() < 100
    underlying = vault.denomination_token
    assert 200 < underlying.fetch_balance_of(vault.address) == 300
