"""Test Enzyme vault deposits are correctly read."""
import datetime
from _decimal import Decimal

import pytest
from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.balance_update import BalanceUpdatePositionType, BalanceUpdateType
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State



def test_enzyme_no_deposit(
    web3: Web3,
    deployer: HexAddress,
    enzyme_vault_contract: Contract,
    vault_comptroller_contract: Contract,
    usdc: Contract,
    user_1: HexAddress,
):
    """Do empty deposit scan."""

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
    assert len(treasury.processed_events) == 0
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
    assert len(treasury.processed_events) == 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None

    # We can scan empty twice
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 0

    # Make a deposit
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault_comptroller_contract.address, 500 * 10**6).transact({"from": user_1})
    vault_comptroller_contract.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})

    # One deposit detected
    events = sync_model.sync_treasury(cycle, state)
    assert len(events) == 1

    # Event was correctly translated
    evt = events[0]
    assert evt.asset.token_symbol == "USDC"
    assert evt.asset.internal_id is None
    assert evt.block_mined_at is not None
    assert evt.past_quantity == Decimal(0)
    assert evt.new_quantity == Decimal(500)
    assert evt.quantity == Decimal(500)
    assert evt.owner_address == user_1
    assert evt.tx_hash is not None
    assert evt.tx_hash is not None
    assert evt.balance_update_id == 2
    assert evt.position_type == BalanceUpdatePositionType.reserve
    assert evt.type == BalanceUpdateType.deposit

    # Sync stat look correct
    assert treasury.last_cycle_at == cycle
    assert treasury.last_updated_at is not None
    assert treasury.last_block_scanned > 1
    assert len(treasury.processed_events) == 1

    # We have one deposit
    assert len(list(treasury.get_deposits())) == 1

    # Strategy has balance
    assert state.portfolio.get_total_equity() == Decimal(500)

    # See we can serialise the sync state
    dump = state.to_json()
    state2: State = State.from_json(dump)
    assert len(state2.sync.treasury.processed_events) == 1


def test_enzyme_two_deposits(
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

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.transfer(user_2, 700 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault_comptroller_contract.address, 500 * 10**6).transact({"from": user_1})
    usdc.functions.approve(vault_comptroller_contract.address, 700 * 10**6).transact({"from": user_2})
    vault_comptroller_contract.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    vault_comptroller_contract.functions.buyShares(700 * 10**6, 1).transact({"from": user_2})

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
    assert reserve_position.get_current_value() == 1200

    # Strategy has its reserve balances updated
    assert state.portfolio.get_total_equity() == pytest.approx(1200)
