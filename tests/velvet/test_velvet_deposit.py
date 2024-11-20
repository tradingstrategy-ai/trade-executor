"""Test Velvet vault deposits are correctly read."""
import datetime
from _decimal import Decimal
from logging import Logger

import pytest
from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from eth_defi.trace import assert_transaction_success_with_explanation

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.balance_update import BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State



def test_velvet_initial_deposit(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
):
    """Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury
    """

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
        reserve_asset=base_usdc,
    )

    state = State()
    sync_model.sync_initial(state)

    # No events yet, because of no deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1

    treasury = state.sync.treasury
    assert treasury.last_block_scanned > 0
    assert len(treasury.balance_update_refs) == 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None

    # We can scan empty twice
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0

    # One deposit detected
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0


def test_enzyme_single_deposit(
    web3: Web3,
    deployer: HexAddress,
    enzyme_vault_contract: Contract,
    vault_comptroller_contract: Contract,
    usdc: Contract,
    user_1: HexAddress,
):
    """Check that the Enzyme sync model can handle a deposit."""

    reorg_mon = create_reorganisation_monitor(web3)

    sync_model = EnzymeVaultSyncModel(
        web3,
        enzyme_vault_contract.address,
        reorg_mon,
    )

    reserve_assert = sync_model.fetch_vault_reserve_asset()
    assert reserve_assert.token_symbol == "USDC"
    assert reserve_assert.decimals == 6

    state = State()
    sync_model.sync_initial(state)

    # No events yet, because of no deposits
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0

    treasury = state.sync.treasury
    assert treasury.last_block_scanned > 0
    assert len(treasury.balance_update_refs) == 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None

    # We can scan empty twice
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0

    # Make a deposit
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault_comptroller_contract.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault_comptroller_contract.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # One deposit detected
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1
    assert state.portfolio.next_balance_update_id == 2

    # Event was correctly translated
    evt = events[0]
    assert evt.asset.token_symbol == "USDC"
    assert evt.asset.internal_id is None
    assert evt.block_mined_at is not None
    assert evt.old_balance == Decimal(0)
    assert evt.quantity == Decimal(500)
    assert evt.owner_address == user_1
    assert evt.tx_hash is not None
    assert evt.tx_hash is not None
    assert evt.balance_update_id == 1
    assert evt.position_type == BalanceUpdatePositionType.reserve
    assert evt.cause == BalanceUpdateCause.deposit

    # Sync stat look correct
    assert treasury.last_cycle_at == cycle
    assert treasury.last_updated_at is not None
    assert treasury.last_block_scanned > 1
    assert len(treasury.balance_update_refs) == 1

    # We have one deposit
    assert len(treasury.balance_update_refs) == 1

    # Strategy has balance
    assert state.portfolio.get_total_equity() == Decimal(500)

    # See we can serialise the sync state
    dump = state.to_json()
    state2: State = State.from_json(dump)
    assert len(state2.sync.treasury.balance_update_refs) == 1


def test_enzyme_two_deposits(
    logger: Logger,
    web3: Web3,
    deployer: HexAddress,
    enzyme_vault_contract: Contract,
    vault_comptroller_contract: Contract,
    usdc: Contract,
    usdc_asset: AssetIdentifier,
    user_1: HexAddress,
    user_2: HexAddress,
):
    """Check that the Enzyme sync model can handle a deposit."""

    reorg_mon = create_reorganisation_monitor(web3)

    sync_model = EnzymeVaultSyncModel(
        web3,
        enzyme_vault_contract.address,
        reorg_mon,
    )

    logger.info("Initial scan")

    state = State()
    sync_model.sync_initial(state)

    logger.info("Broadcast")

    # Make two deposits from separate parties
    txs = set()
    txs.add(usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer}))
    txs.add(usdc.functions.transfer(user_2, 700 * 10**6).transact({"from": deployer}))
    txs.add(usdc.functions.approve(vault_comptroller_contract.address, 500 * 10**6).transact({"from": user_1}))
    txs.add(usdc.functions.approve(vault_comptroller_contract.address, 700 * 10**6).transact({"from": user_2}))
    txs.add(vault_comptroller_contract.functions.buyShares(500 * 10**6, 1).transact({"from": user_1}))
    txs.add(vault_comptroller_contract.functions.buyShares(700 * 10**6, 1).transact({"from": user_2}))

    for tx_hash in txs:
        logger.info("Confirming tx %s", tx_hash)
        assert_transaction_success_with_explanation(web3, tx_hash)

    logger.info("Deposit scan")

    # One deposit detected
    cycle = datetime.datetime.utcnow()
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 2

    # The strategy treasury has been
    reserve_position = state.portfolio.get_reserve_position(usdc_asset)
    assert reserve_position.last_sync_at is not None
    assert reserve_position.last_pricing_at is not None
    assert reserve_position.quantity == 1200
    assert reserve_position.reserve_token_price == 1
    assert reserve_position.get_value() == 1200

    # Strategy has its reserve balances updated
    assert state.portfolio.get_total_equity() == pytest.approx(1200)
