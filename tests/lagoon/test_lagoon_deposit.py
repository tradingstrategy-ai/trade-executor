"""Test Velvet vault deposits are correctly read."""
import datetime
from decimal import Decimal

import pytest
from eth_typing import HexAddress

from eth_defi.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.portfolio import ReserveMissing
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


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


def test_lagoon_deposit(
    automated_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
):
    """Do the initial lagoon deposit."""

    vault = automated_lagoon_vault.vault
    usdc = base_usdc
    strategy_universe = vault_strategy_universe

    sync_model = LagoonVaultSyncModel(
        vault=vault,
        hot_wallet=None,
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
    events = sync_model.sync_treasury(cycle, state)
    treasury = state.sync.treasury
    reserve_position = state.portfolio.get_default_reserve_position()
    assert len(events) == 1
    assert treasury.last_block_scanned > 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1

    # We scan again, no changes
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0
    assert len(treasury.balance_update_refs) == 1
    assert len(reserve_position.balance_updates) == 1

    # Check we account USDC correctly
    portfolio = state.portfolio
    assert portfolio.get_cash() == pytest.approx(9)
