import os

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.vault.base import VaultSpec
from eth_defi.velvet import VelvetVault

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE", "https://mainnet.base.org")


@pytest.fixture()
def vault_owner() -> HexAddress:
    # Vaut owner
    return "0x0c9db006f1c7bfaa0716d70f012ec470587a8d4f"


@pytest.mark.skipif(not JSON_RPC_BASE, reason="JSON_RPC_BASE is needed to run mainnet fork tets")
@pytest.fixture()
def anvil_base_fork(request, vault_owner) -> AnvilLaunch:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        unlocked_addresses=[vault_owner],
    )
    try:
        yield launch
    finally:
        # Wind down Anvil process after the test is complete
        launch.close()


@pytest.fixture()
def web3(anvil_base_fork) -> Web3:
    web3 = create_multi_provider_web3(anvil_base_fork.json_rpc_url)
    assert web3.eth.chain_id == 8453
    return web3


@pytest.fixture(scope='module')
def base_test_vault_spec() -> VaultSpec:
    """Vault https://dapp.velvet.capital/ManagerVaultDetails/0x205e80371f6d1b33dff7603ca8d3e92bebd7dc25"""
    return VaultSpec(1, "0x205e80371f6d1b33dff7603ca8d3e92bebd7dc25")


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
    return "0x6921B130D297cc43754afba22e5EAc0FBf8Db75b"


@pytest.fixture()
def base_example_vault(web3, base_test_vault_spec: VaultSpec) -> VelvetVault:
    return VelvetVault(web3, base_test_vault_spec)


@pytest.fixture(scope='module')
def base_usdc(base_test_reserve_asset) -> AssetIdentifier:
    """USDC"""
    return AssetIdentifier(
        chain_id=8453,
        address=base_test_reserve_asset,
        decimals=6,
        token_symbol="USDC",
    )

@pytest.fixture(scope='module')
def base_doginme(base_test_volatile_asset) -> AssetIdentifier:
    """DogInMe"""
    return AssetIdentifier(
        chain_id=8453,
        address=base_test_volatile_asset,
        decimals=18,
        token_symbol="DogInMe",
    )



@pytest.fixture(scope='module')
def velvet_test_vault_pair_universe(
    base_usdc,
    base_doginme,
) -> PandasPairUniverse:
    """Define pair universe of USDC and DogMeIn assets, trading on Uni v3 on Base.

    """

    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(8453),
                chain_slug="base",
                exchange_id=1,
                exchange_slug="uniswap-v3",
                address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
                exchange_type=ExchangeType.uniswap_v3,
                pair_count=1,
            )
        }
    )

    # USDC pool has zero liquidity on Base
    # https://app.uniswap.org/explore/tokens/base/0x6921b130d297cc43754afba22e5eac0fbf8db75b
    # https://app.uniswap.org/explore/pools/base/0x386298ce505067CA53e8a70FE82E12ff1dA7cc38

    # https://www.coingecko.com/en/coins/doginme
    # https://app.uniswap.org/explore/tokens/base/0x6921b130d297cc43754afba22e5eac0fbf8db75b
    trading_pair = TradingPairIdentifier(
        base=base_doginme,
        quote=base_usdc,
        pool_address="0x386298ce505067CA53e8a70FE82E12ff1dA7cc38",
        # https://docs.uniswap.org/contracts/v3/reference/deployments/base-deployments
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
    )

    universe = create_universe_from_trading_pair_identifiers(
        [trading_pair],
        exchange_universe=exchange_universe,
    )
    return universe

