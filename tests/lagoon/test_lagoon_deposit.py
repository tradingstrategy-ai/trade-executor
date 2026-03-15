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
from eth_defi.compat import native_datetime_utc_now


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
    cycle = native_datetime_utc_now()
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
    cycle = native_datetime_utc_now()
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
    cycle = native_datetime_utc_now()
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


def test_lagoon_nav_with_exchange_account_position(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    vault_strategy_universe: TradingStrategyUniverse,
    depositor: HexAddress,
    another_new_depositor: HexAddress,
    asset_manager: HotWallet,
):
    """Verify that NAV calculation uses on-chain reserves, not stale portfolio reserves.

    Reproduces a production bug where:

    1. USDC is deposited into the vault and settled (reserves synced to 25)
    2. USDC is transferred from the Safe to an exchange (simulating GMX sendTokens)
    3. An exchange account position is created (simulating correct-accounts)
    4. The portfolio double-counts: stale reserves + exchange account value
    5. A second deposit is priced at the inflated NAV, harming the depositor

    The fix reconciles reserves from the on-chain Safe balance before
    calculating NAV, preventing the double-counting.

    See README-GMX-Lagoon.md for the full GMX token flow.
    """
    from tradeexecutor.exchange_account.state import open_exchange_account_position
    from tradeexecutor.state.identifier import TradingPairIdentifier, TradingPairKind, AssetIdentifier
    from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType

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

    # Step 1: Initial deposit of 25 USDC and settle
    usdc_amount = Decimal(25)
    raw_usdc_amount = usdc.convert_to_raw(usdc_amount)
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, raw_usdc_amount)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    cycle = native_datetime_utc_now()
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)
    assert len(events) == 1

    portfolio = state.portfolio
    assert portfolio.get_cash() == pytest.approx(25)
    initial_nav = sync_model.calculate_valuation(state)
    assert initial_nav == pytest.approx(25)

    # Record initial share price
    block = web3.eth.block_number
    initial_total_assets = float(vault.fetch_total_assets(block))
    initial_total_supply = float(vault.fetch_total_supply(block))
    initial_share_price = initial_total_assets / initial_total_supply

    # Step 2: Transfer 20 USDC out of Safe (simulating GMX sendTokens)
    safe_address = vault.safe_address
    burn_address = web3.eth.accounts[9]
    web3.provider.make_request("anvil_impersonateAccount", [safe_address])
    tx_hash = usdc.transfer(burn_address, Decimal(20)).transact({"from": safe_address, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    web3.provider.make_request("anvil_stopImpersonatingAccount", [safe_address])

    # Verify Safe USDC dropped but portfolio reserves are still stale
    safe_usdc = usdc.fetch_balance_of(safe_address)
    assert safe_usdc == pytest.approx(Decimal(5))
    assert portfolio.get_cash() == pytest.approx(25)  # stale!

    # Step 3: Create exchange account position (simulating correct-accounts)
    exchange_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=8453,
            address="0x0000000000000000000000000000000000000001",
            token_symbol="EXCH-ACCOUNT",
            decimals=6,
        ),
        quote=usdc_asset,
        pool_address="0x0000000000000000000000000000000000000001",
        exchange_address="0x0000000000000000000000000000000000000001",
        internal_id=999,
        internal_exchange_id=999,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
    )

    open_exchange_account_position(
        state=state,
        strategy_cycle_at=cycle,
        pair=exchange_pair,
        reserve_currency=usdc_asset,
        reserve_amount=Decimal(1),  # placeholder, like correct-accounts
        notes="Test exchange account position",
    )

    # Verify reserve deduction from open_exchange_account_position
    assert portfolio.get_cash() == pytest.approx(24)  # 25 - 1 deducted

    # Step 4: Set exchange account position value to ~20 (simulating API sync)
    exchange_pos = list(portfolio.open_positions.values())[-1]
    assert exchange_pos.is_exchange_account()

    now = native_datetime_utc_now()
    evt = BalanceUpdate(
        balance_update_id=1,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.vault_flow,
        asset=exchange_pair.base,
        block_mined_at=now,
        strategy_cycle_included_at=cycle,
        chain_id=8453,
        old_balance=Decimal(1),
        usd_value=19.0,
        quantity=Decimal(19),
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=exchange_pos.position_id,
        block_number=None,
        notes="Simulated exchange account valuation",
    )
    exchange_pos.balance_updates[evt.balance_update_id] = evt
    exchange_pos.last_pricing_at = now
    exchange_pos.last_token_price = 1.0

    # Step 5: Verify the double-counting pre-condition
    # Without the fix, NAV would be: stale 24 + position 20 = 44 (wrong!)
    assert portfolio.get_net_asset_value() == pytest.approx(44, abs=1)

    # Step 6: sync_treasury reconciles reserves from on-chain
    events = sync_model.sync_treasury(cycle, state, post_valuation=True)

    reserve_position = portfolio.get_default_reserve_position()
    # After reconciliation, reserves match on-chain Safe balance
    assert reserve_position.quantity == pytest.approx(Decimal(5), abs=Decimal(1))
    # NAV is now correct: 5 (on-chain cash) + 20 (position) = 25
    nav_after = sync_model.calculate_valuation(state)
    assert nav_after == pytest.approx(25, abs=1)

    # Step 7: Second deposit — verify share price is preserved
    block_before = web3.eth.block_number
    total_assets_before = float(vault.fetch_total_assets(block_before))
    total_supply_before = float(vault.fetch_total_supply(block_before))
    share_price_before = total_assets_before / total_supply_before

    usdc_deposit_2 = Decimal(100)
    raw_deposit_2 = usdc.convert_to_raw(usdc_deposit_2)
    tx_hash = usdc.approve(vault.address, usdc_deposit_2).transact({"from": another_new_depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(another_new_depositor, raw_deposit_2)
    tx_hash = deposit_func.transact({"from": another_new_depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    cycle2 = native_datetime_utc_now()
    events = sync_model.sync_treasury(cycle2, state, post_valuation=True)
    assert len(events) == 1

    # Finalise deposit so shares are in the depositor's wallet
    bound_func = vault.finalise_deposit(another_new_depositor)
    tx_hash = bound_func.transact({"from": another_new_depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Check minted shares are fair — should be ~100 / share_price
    depositor_shares = vault.share_token.fetch_balance_of(another_new_depositor)
    expected_shares = Decimal(str(usdc_deposit_2)) / Decimal(str(share_price_before))
    assert depositor_shares == pytest.approx(expected_shares, rel=0.02)

    # Check share price is preserved after deposit
    block_after = web3.eth.block_number
    total_assets_after = float(vault.fetch_total_assets(block_after))
    total_supply_after = float(vault.fetch_total_supply(block_after))
    share_price_after = total_assets_after / total_supply_after

    assert share_price_after == pytest.approx(share_price_before, rel=0.01)