"""Test Velvet vault deposits are correctly read."""
import datetime

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.velvet import VelvetVault

from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import ReserveMissing
from tradeexecutor.state.state import State







def test_velvet_treasury_initialise(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
):
    """Initialise Velvet treasury

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury
    """

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
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
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    assert len(portfolio.reserves) == 1  # USDC added as the reserve asset
    assert len(treasury.balance_update_refs) == 0  # No deposits processed yet

    # We have reserve position now, but without any balance
    reserve_position = portfolio.get_default_reserve_position()
    assert len(reserve_position.balance_updates) == 0  # No deposits processed yet
    assert reserve_position.asset.get_identifier() == "8453-0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
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
    assert portfolio.get_cash() == pytest.approx(2.674828)


def test_velvet_deposit(
    web3: Web3,
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    deposit_user,
    base_test_reserve_asset: HexAddress,
):
    """Sync Velvet deposit"""

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    assert len(portfolio.reserves) == 1  # USDC added as the reserve asset

    # Process the initial deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1
    assert portfolio.get_cash() == pytest.approx(2.674828)

    #
    # Process additional deposits
    #

    # Prepare the deposit tx payload
    tx_data = sync_model.vault.prepare_deposit_with_enso(
        from_=deposit_user,
        deposit_token_address=base_test_reserve_asset,
        amount=5 * 10**6,
    )
    tx_hash = web3.eth.send_transaction(tx_data)
    assert_transaction_success_with_explanation(web3, tx_hash)

