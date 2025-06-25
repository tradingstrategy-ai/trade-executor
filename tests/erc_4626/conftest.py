"""Vault trading fixtures.

- Set up a trading universe on Base mainnet fork using one IPOR vault
"""
import os
from decimal import Decimal

import pytest
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.hotwallet import HotWallet
from eth_defi.ipor.vault import IPORVault
from eth_defi.lagoon.deployment import deploy_automated_lagoon_vault, LagoonDeploymentParameters, LagoonAutomatedDeployment
from eth_defi.lagoon.vault import LagoonVault
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import TokenDetails, fetch_erc20_details, USDC_NATIVE_TOKEN
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.constants import UNISWAP_V2_DEPLOYMENTS
from eth_defi.uniswap_v2.deployment import fetch_deployment, UniswapV2Deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.vault.base import VaultSpec
from eth_defi.uniswap_v3.deployment import fetch_deployment as fetch_deployment_uni_v3, UniswapV3Deployment

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

CI = os.environ.get("CI", None) is not None

pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


@pytest.fixture()
def usdc_holder() -> HexAddress:
    # https://basescan.org/token/0x833589fcd6edb6e08f4c7c32d4f71b54bda02913#balances
    return "0x3304E22DDaa22bCdC5fCa2269b418046aE7b566A"


@pytest.fixture()
def test_block_number() -> int:
    """When we fork"""
    return 27975506


@pytest.fixture()
def vault(web3) -> IPORVault:
    # https://app.ipor.io/fusion/base/0x45aa96f0b3188d47a1dafdbefce1db6b37f58216
    spec = VaultSpec(8545, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216")
    return IPORVault(web3, spec, features={ERC4626Feature.ipor_like})


@pytest.fixture()
def anvil_base_fork(request, usdc_holder, test_block_number) -> AnvilLaunch:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """
    assert JSON_RPC_BASE, "JSON_RPC_BASE not set"
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        unlocked_addresses=[usdc_holder],
        fork_block_number=test_block_number,
    )
    try:
        yield launch
    finally:
        # Wind down Anvil process after the test is complete
        launch.close()


@pytest.fixture()
def web3(anvil_base_fork) -> Web3:
    """Create a web3 connector.

    - By default use Anvil forked Base

    - Eanble Tenderly testnet with `JSON_RPC_TENDERLY` to debug
      otherwise impossible to debug Gnosis Safe transactions
    """

    tenderly_fork_rpc = os.environ.get("JSON_RPC_TENDERLY", None)

    if tenderly_fork_rpc:
        web3 = create_multi_provider_web3(tenderly_fork_rpc)
    else:
        web3 = create_multi_provider_web3(
            anvil_base_fork.json_rpc_url,
            default_http_timeout=(3, 250.0),  # multicall slow, so allow improved timeout
        )
    assert web3.eth.chain_id == 8453
    return web3


@pytest.fixture(scope='module')
def base_weth() -> AssetIdentifier:
    """WETH"""
    return AssetIdentifier(
        chain_id=8453,
        address="0x4200000000000000000000000000000000000006",
        decimals=18,
        token_symbol="WETH",
    )


@pytest.fixture(scope='module')
def base_usdc() -> AssetIdentifier:
    """USDC"""
    return AssetIdentifier(
        chain_id=8453,
        address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        decimals=6,
        token_symbol="USDC",
    )


@pytest.fixture()
def base_usdc_token(base_usdc, web3) -> TokenDetails:
    """USDC"""
    return fetch_erc20_details(
        web3,
        base_usdc.address,
        chain_id=base_usdc.chain_id,
    )


@pytest.fixture()
def hot_wallet(web3, usdc_holder) -> HotWallet:
    """A test account with USDC balance."""

    hw = HotWallet.create_for_testing(
        web3,
        test_account_n=1,
        eth_amount=10
    )
    hw.sync_nonce(web3)

    # give hot wallet some native token
    web3.eth.send_transaction(
        {
            "from": web3.eth.accounts[9],
            "to": hw.address,
            "value": 1 * 10**18,
        }
    )

    usdc = fetch_erc20_details(web3, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")

    # Top up with 999 USDC
    tx_hash = usdc.contract.functions.transfer(hw.address, 999 * 10**6).transact({"from": usdc_holder, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return hw


@pytest.fixture()
def execution_model(
    web3,
    hot_wallet: HotWallet,
) -> EthereumExecution:
    """Set EthereumExecutionModel in mainnet fork testing mode."""
    execution_model = EthereumExecution(
        HotWalletTransactionBuilder(web3, hot_wallet),
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    return execution_model


@pytest.fixture()
def routing_model(execution_model, strategy_universe) -> GenericRouting:
    return execution_model.create_default_routing_model(strategy_universe)


@pytest.fixture()
def pricing_model(
    web3,
    strategy_universe,
) -> GenericPricing:
    pair_configurator = EthereumPairConfigurator(
        web3,
        strategy_universe,
    )

    weth_usdc = strategy_universe.get_pair_by_human_description(
        (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
    )

    base_weth = weth_usdc.base

    return GenericPricing(
        pair_configurator,
        exchange_rate_pairs={
            base_weth: weth_usdc,
        }
    )


@pytest.fixture()
def ipor_usdc(vault: IPORVault) -> TradingPairIdentifier:
    ipor_usdc = translate_vault_to_trading_pair(vault)
    assert ipor_usdc.is_vault()
    return ipor_usdc


@pytest.fixture()
def vault_pair_universe(
    vault: IPORVault,
    base_usdc,
    base_weth,
    ipor_usdc,
) -> PandasPairUniverse:
    """Define pair universe with some DEX pairs and and a vault."""

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
            ),

            2: Exchange(
                chain_id=ChainId(8453),
                chain_slug="base",
                exchange_id=2,
                exchange_slug="uniswap-v2",
                address="0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",
                exchange_type=ExchangeType.uniswap_v2,
                pair_count=1,
            ),

            3: Exchange(
                chain_id=ChainId(8453),
                chain_slug="base",
                exchange_id=3,
                exchange_slug="ipor",
                address="0x0000000000000000000000000000000000000000",
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )

    weth_usdc_uniswap_v3 = TradingPairIdentifier(
        base=base_weth,
        quote=base_usdc,
        pool_address="0xd0b53d9277642d899df5c87a3966a349a798f224",
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        fee=0.0005,
    )

    universe = create_universe_from_trading_pair_identifiers(
        [
            weth_usdc_uniswap_v3,
            ipor_usdc,
        ],
        exchange_universe=exchange_universe,
    )
    return universe


@pytest.fixture()
def strategy_universe(
    vault_pair_universe,
) -> TradingStrategyUniverse:
    """Contains pairs and exchanges"""

    universe = Universe(
        chains={ChainId.base},
        time_bucket=TimeBucket.not_applicable,  # Not used, only live price checks done
        exchange_universe=vault_pair_universe.exchange_universe,
        pairs=vault_pair_universe,
    )

    usdc = vault_pair_universe.get_token("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets={translate_token(usdc)},
    )


@pytest.fixture()
def sync_model(
    web3,
    hot_wallet,
) -> HotWalletSyncModel:
    sync_model = HotWalletSyncModel(web3, hot_wallet)
    return sync_model
