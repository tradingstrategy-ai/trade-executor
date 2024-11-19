import pytest
from eth_typing import HexAddress

from eth_defi.vault.base import VaultSpec
from eth_defi.velvet import VelvetVault


@pytest.fixture(scope='module')
def base_test_vault_spec() -> VaultSpec:
    """Vault https://dapp.velvet.capital/ManagerVaultDetails/0x205e80371f6d1b33dff7603ca8d3e92bebd7dc25"""
    return VaultSpec(1, "0x205e80371f6d1b33dff7603ca8d3e92bebd7dc25")


@pytest.fixture(scope='module')
def base_test_reserve_asset() -> HexAddress:
    """USDC"""
    return "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"


@pytest.fixture(scope='module')
def base_test_volatile_asset() -> HexAddress:
    """DogInMe"""
    return "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"


@pytest.fixture(scope='module')
def base_example_vault(web3, base_test_vault_spec: VaultSpec) -> VelvetVault:
    return VelvetVault(web3, base_test_vault_spec)