"""Test Lagoon vault deposits/redemptions are correctly performed."""
import datetime
import os
from decimal import Decimal

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.portfolio import ReserveMissing
from tradeexecutor.state.state import State
from tradeexecutor.statistics.core import calculate_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


def test_lagoon_treasury_initialise(
    automated_lagoon_vault: LagoonAutomatedDeployment,
    vault_strategy_universe: TradingStrategyUniverse,
):
    """Initialise the treasury"""

    vault = automated_lagoon_vault.vault
    asset_usdc = vault_strategy_universe.get_reserve_asset()

    sync_model = LagoonVaultSyncModel(
        vault=vault,
        hot_wallet=None,
    )

    state = State()
    treasury = state.sync.treasury
    portfolio = state.portfolio
    assert len(portfolio.reserves) == 0
    assert len(treasury.balance_update_refs) == 0
    with pytest.raises(ReserveMissing):
        portfolio.get_default_reserve_position()

    sync_model.sync_initial(
        state,
        reserve_asset=asset_usdc,
        reserve_token_price=1.0,
    )
    assert len(portfolio.reserves) == 1  # USDC added as the reserve asset
    assert len(treasury.balance_update_refs) == 0  # No deposits processed yet

    # We have reserve position now, but without any balance
    reserve_position = portfolio.get_default_reserve_position()
    assert len(reserve_position.balance_updates) == 0  # No deposits processed yet
    assert reserve_position.asset.get_identifier() == "8453-0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

    # No initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0
    assert treasury.last_block_scanned is None
    assert treasury.last_updated_at is None
    assert treasury.last_cycle_at is None
    assert len(treasury.balance_update_refs) == 0
    assert len(reserve_position.balance_updates) == 0


def test_lagoon_sync_deposit(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
    asset_manager: HotWallet,
):
    """Do a deposit to Lagoon vault and sync it."""

    vault = automated_lagoon_vault.vault
    usdc = base_usdc_token
    strategy_universe = vault_strategy_universe

    sync_model = LagoonVaultSyncModel(
        vault=vault,
        hot_wallet=asset_manager,
    )

    state = State()
    usdc_asset = strategy_universe.get_reserve_asset()
    sync_model.sync_initial(
        state,
        reserve_asset=usdc_asset,
        reserve_token_price=1.0,
    )

    # Do initial deposit
    # Deposit 9.00 USDC into the vault from the first user
    usdc_amount = Decimal(9.00)
    raw_usdc_amount = usdc.convert_to_raw(usdc_amount)
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, raw_usdc_amount)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.fetch_balance_of(vault.silo_address) == pytest.approx(Decimal(9))

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)
    treasury = state.sync.treasury
    reserve_position = state.portfolio.get_default_reserve_position()
    assert len(events) == 1
    assert treasury.last_block_scanned > 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1
    deposit_event = events[0]
    assert deposit_event.block_number is not None
    assert deposit_event.tx_hash is not None
    assert deposit_event.other_data is not None
    assert deposit_event.quantity == Decimal(9)
    assert deposit_event.old_balance == Decimal(0)
    assert deposit_event.get_share_count() == Decimal(9)

    # Check we have share price recorded
    treasury = state.sync.treasury
    assert treasury.share_count == 9

    # We scan again, no changes
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1

    # Check we account USDC correctly
    portfolio = state.portfolio
    assert portfolio.get_cash() == pytest.approx(9)
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(9)
    assert reserve_position.last_sync_at is not None
    assert reserve_position.last_pricing_at is not None

    # Check we calculate share price for statistics
    statistics = calculate_statistics(
        clock=cycle,
        portfolio=portfolio,
        execution_mode=ExecutionMode.unit_testing,
        treasury=treasury,
    )
    assert statistics.portfolio.share_price_usd == pytest.approx(1.0)
    assert statistics.portfolio.share_count == Decimal(9)



def test_lagoon_sync_redeem(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
    asset_manager: HotWallet,
):
    """Do a redeem to Lagoon vault and sync it."""

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
    # Deposit 9.00 USDC into the vault from the first user
    usdc_amount = Decimal(9.00)
    raw_usdc_amount = usdc.convert_to_raw(usdc_amount)
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, raw_usdc_amount)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.fetch_balance_of(vault.silo_address) == pytest.approx(Decimal(9))

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)
    assert len(events) == 1

    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(9)

    #
    # Redeem
    #

    # Get shares to the wallet
    bound_func = vault.finalise_deposit(depositor)
    tx_hash = bound_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Redeem shares
    # Redeem 3 USDC for the first user
    shares_to_redeem_raw = vault.share_token.convert_to_raw(3)
    bound_func = vault.request_redeem(depositor, shares_to_redeem_raw)
    tx_hash = bound_func.transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Sync
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)
    treasury = state.sync.treasury
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.quantity == pytest.approx(6)
    assert len(events) == 1
    assert len(treasury.balance_update_refs) == 2
    assert len(reserve_position.balance_updates) == 2
    redeem_event = events[0]
    assert redeem_event.block_number is not None
    assert redeem_event.tx_hash is not None
    assert redeem_event.other_data is not None
    assert redeem_event.quantity == pytest.approx(Decimal(-3))
    assert redeem_event.old_balance == pytest.approx(Decimal(9))

    assert portfolio.get_cash() == pytest.approx(pytest.approx(6))