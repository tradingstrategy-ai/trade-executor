"""Test Enzyme vault deposits are correctly read."""
import datetime

from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.state import State



def test_enzyme_deposit(
    web3: Web3,
    deployer: HexAddress,
    enzyme_vault_contract: Contract,
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
    events = sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert len(events) == 0

    treasury = state.sync.treasury
    assert treasury.last_block_scanned > 0
    assert len(treasury.processed_events) == 0
    assert treasury.last_updated_at is not None
    assert treasury.last_cycle_at is not None

    # Make a deposit











