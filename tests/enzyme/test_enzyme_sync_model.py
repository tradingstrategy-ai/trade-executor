"""Test Enzyme sync model."""
from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.enzyme.deployment import EnzymeDeployment, RateAsset

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.state import State


def test_enzyme_sync_model_init(
    web3: Web3,
    deployer: HexAddress,
    user_1: HexAddress,
    user_2: HexAddress,
    usdc: Contract,
    weth: Contract,
    mln: Contract,
    usdc_usd_mock_chainlink_aggregator: Contract,
):
    """Fetch the initial data vault data from the chain."""

    deployment = EnzymeDeployment.deploy_core(
        web3,
        deployer,
        mln,
        weth,
    )

    # Create a vault for user 1
    # where we nominate everything in USDC
    deployment.add_primitive(
        usdc,
        usdc_usd_mock_chainlink_aggregator,
        RateAsset.USD,
    )

    comptroller_contract, vault_contract = deployment.create_new_vault(
        user_1,
        usdc,
    )

    sync_model = EnzymeVaultSyncModel(web3, vault_contract.address)
    state = State()
    sync_model.sync_initial(state)

    deployment = state.sync.deployment
    assert deployment.address == vault_contract.address
    assert deployment.block_number > 0
    assert deployment.tx_hash is not None
    assert deployment.block_mined_at is not None
    assert deployment.vault_token_symbol == "EXAMPLE"
    assert deployment.vault_token_name == "Example Fund"
