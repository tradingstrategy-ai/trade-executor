"""Test Enzyme sync model."""
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradingstrategy.chain import ChainId

from tradeexecutor.state.state import State


def test_velvet_sync_model_init(
    base_example_vault: VelvetVault,
):
    """Fetch the initial data vault data from the chain."""

    vault = base_example_vault

    sync_model = VelvetVaultSyncModel(
        vault,
        hot_wallet=None,
    )
    state = State()
    sync_model.sync_initial(state)

    deployment = state.sync.deployment
    assert deployment.address == vault.vault_address
    assert deployment.block_number > 0
    assert deployment.tx_hash is not None
    assert deployment.block_mined_at is not None
    assert deployment.vault_token_symbol == "EXAMPLE"
    assert deployment.vault_token_name == "Example Fund"
    assert deployment.chain_id == ChainId.anvil
