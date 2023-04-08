"""Test Enzyme sync model."""
from tradingstrategy.chain import ChainId
from web3 import Web3

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.state import State


def test_enzyme_sync_model_init(
    web3: Web3,
    enzyme_vault_contract,
):
    """Fetch the initial data vault data from the chain."""

    reorg_mon = create_reorganisation_monitor(web3)

    sync_model = EnzymeVaultSyncModel(web3, enzyme_vault_contract.address, reorg_mon)
    state = State()
    sync_model.sync_initial(state)

    deployment = state.sync.deployment
    assert deployment.address == enzyme_vault_contract.address
    assert deployment.block_number > 0
    assert deployment.tx_hash is not None
    assert deployment.block_mined_at is not None
    assert deployment.vault_token_symbol == "EXAMPLE"
    assert deployment.vault_token_name == "Example Fund"
    assert deployment.chain_id == ChainId.anvil
