
"""Arbitrum Sepolia fork based tests for Orderly.

Tests are based on the Orderly vault deployment on Arbitrum Sepolia testnet.
"""
import os

import pytest
from eth_account import Account
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.vault.base import VaultSpec

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.orderly.orderly_execution import OrderlyExecution
from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.ethereum.orderly.orderly_routing import OrderlyRouting
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
HOT_WALLET_PRIVATE_KEY = os.environ.get("HOT_WALLET_PRIVATE_KEY")

CI = os.environ.get("CI", None) is not None

pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")


@pytest.fixture()
def large_usdc_holder() -> HexAddress:
    """A known address with large USDC balance on Arbitrum Sepolia.
    
    This needs to be updated if the fork block changes.
    """
    # You may need to find an appropriate holder for the test
    return "0x0000000000000000000000000000000000000000"  # TODO: Find actual USDC holder on Arbitrum Sepolia


@pytest.fixture()
def anvil_arbitrum_sepolia_fork(request, large_usdc_holder) -> AnvilLaunch:
    """Create a testable fork of live Arbitrum Sepolia chain.

    :return: JSON-RPC URL for Web3
    """
    assert JSON_RPC_ARBITRUM_SEPOLIA, "JSON_RPC_ARBITRUM_SEPOLIA not set"
    launch = fork_network_anvil(
        JSON_RPC_ARBITRUM_SEPOLIA,
        unlocked_addresses=[large_usdc_holder] if large_usdc_holder != "0x0000000000000000000000000000000000000000" else [],
        fork_block_number=178687280,  # Can be updated to a more recent block
    )
    try:
        yield launch
    finally:
        # Wind down Anvil process after the test is complete
        launch.close()


@pytest.fixture()
def web3(anvil_arbitrum_sepolia_fork) -> Web3:
    """Create a web3 connector.

    - By default use Anvil forked Arbitrum Sepolia

    - Enable Tenderly testnet with `JSON_RPC_TENDERLY` to debug
      otherwise impossible to debug transactions
    """

    tenderly_fork_rpc = os.environ.get("JSON_RPC_TENDERLY", None)

    if tenderly_fork_rpc:
        web3 = create_multi_provider_web3(tenderly_fork_rpc)
    else:
        web3 = create_multi_provider_web3(
            anvil_arbitrum_sepolia_fork.json_rpc_url,
            default_http_timeout=(3, 250.0),  # multicall slow, so allow improved timeout
        )
    assert web3.eth.chain_id == 421614  # Arbitrum Sepolia chain ID
    return web3


@pytest.fixture(scope='module')
def usdc() -> AssetIdentifier:
    """USDC on Arbitrum Sepolia"""
    return AssetIdentifier(
        chain_id=421614,
        address="0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d",
        decimals=6,
        token_symbol="USDC",
    )


@pytest.fixture(scope='module')
def weth() -> AssetIdentifier:
    """WETH on Arbitrum Sepolia"""
    return AssetIdentifier(
        chain_id=421614,
        address="0x7b79995e5f793A07Bc00c21412e50Ecae098E7f9",
        decimals=18,
        token_symbol="WETH",
    )


@pytest.fixture()
def usdc_token(web3, usdc) -> TokenDetails:
    """USDC token details"""
    return fetch_erc20_details(
        web3,
        usdc.address,
        chain_id=usdc.chain_id,
    )


@pytest.fixture()
def weth_token(web3, weth) -> TokenDetails:
    """WETH token details"""
    return fetch_erc20_details(
        web3,
        weth.address,
        chain_id=weth.chain_id,
    )


@pytest.fixture()
def hot_wallet(web3, usdc_token, large_usdc_holder) -> HotWallet:
    """A test account with USDC balance."""
    
    if HOT_WALLET_PRIVATE_KEY:
        # Use provided private key
        hw = HotWallet(Account.from_key(HOT_WALLET_PRIVATE_KEY))
    else:
        # Create a test wallet
        hw = HotWallet.create_for_testing(
            web3,
            test_account_n=1,
            eth_amount=10
        )
    
    hw.sync_nonce(web3)

    # Give hot wallet some native token for gas
    web3.eth.send_transaction(
        {
            "from": web3.eth.accounts[9],
            "to": hw.address,
            "value": 1 * 10**18,
        }
    )

    # Top up with USDC if we have a holder
    if large_usdc_holder != "0x0000000000000000000000000000000000000000":
        tx_hash = usdc_token.contract.functions.transfer(hw.address, 1000 * 10**6).transact({"from": large_usdc_holder, "gas": 100_000})
        assert_transaction_success_with_explanation(web3, tx_hash)
    
    return hw


@pytest.fixture()
def orderly_vault_address() -> HexAddress:
    """Orderly vault address on Arbitrum Sepolia.

    This is the deployed Orderly vault contract address on Arbitrum Sepolia testnet.
    """
    return "0x0EaC556c0C2321BA25b9DC01e4e3c95aD5CDCd2f"


@pytest.fixture()
def broker_id() -> str:
    """Orderly broker ID for testing"""
    return "woofi_pro"


@pytest.fixture()
def orderly_account_id() -> HexAddress:
    """Orderly account ID for testing"""
    # This is a test account ID - should be replaced with actual test account
    return "0xca47e3fb4339d0e30c639bb30cf8c2d18cbb8687a27bc39249287232f86f8d00"


@pytest.fixture()
def orderly_vault(web3, orderly_vault_address, broker_id) -> OrderlyVault:
    """Orderly vault instance"""
    return OrderlyVault(
        web3=web3,
        spec=VaultSpec(
            chain_id=421614,
            vault_address=orderly_vault_address,),
        broker_id=broker_id,
    )


@pytest.fixture()
def asset_manager(web3) -> HotWallet:
    """Account that we use for Orderly trades"""
    hot_wallet = HotWallet.create_for_testing(web3, eth_amount=1)
    return hot_wallet


@pytest.fixture()
def orderly_tx_builder(
    orderly_vault: OrderlyVault,
    asset_manager: HotWallet,
    broker_id: str,
    orderly_account_id: str,
) -> OrderlyTransactionBuilder:
    """Orderly transaction builder instance"""
    return OrderlyTransactionBuilder(
        vault=orderly_vault,
        hot_wallet=asset_manager,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
    )


@pytest.fixture()
def orderly_execution_model(
    web3: Web3,
    orderly_vault: OrderlyVault,
    orderly_tx_builder: OrderlyTransactionBuilder,
    broker_id: str,
    orderly_account_id: str,
) -> OrderlyExecution:
    """Set OrderlyExecution in Arbitrum Sepolia fork testing mode."""

    execution_model = OrderlyExecution(
        vault=orderly_vault,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
        tx_builder=orderly_tx_builder,
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    return execution_model


@pytest.fixture()
def orderly_routing_model(orderly_execution_model, orderly_strategy_universe) -> OrderlyRouting:
    """Create routing model for Orderly"""
    return orderly_execution_model.create_default_routing_model(orderly_strategy_universe)


@pytest.fixture()
def orderly_pricing_model(
    web3,
    orderly_strategy_universe,
) -> GenericPricing:
    """Create pricing model for Orderly"""
    pair_configurator = EthereumPairConfigurator(
        web3,
        orderly_strategy_universe,
    )

    weth_usdc = orderly_strategy_universe.get_pair_by_human_description(
        (ChainId.arbitrum_sepolia, "uniswap-v3", "WETH", "USDC", 0.0005),
    )

    weth = weth_usdc.base

    return GenericPricing(
        pair_configurator,
        exchange_rate_pairs={
            weth: weth_usdc,
        }
    )


@pytest.fixture(scope='module')
def orderly_pair_universe(
    usdc,
    weth,
) -> PandasPairUniverse:
    """Define pair universe for testing on Arbitrum Sepolia.
    
    This is a simplified universe with WETH-USDC pair for testing.
    """

    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(421614),  # Arbitrum Sepolia
                chain_slug="arbitrum-sepolia",
                exchange_id=1,
                exchange_slug="uniswap-v3",
                address="0x0000000000000000000000000000000000000000",  # TODO: Add actual Uniswap v3 factory on Arbitrum Sepolia
                exchange_type=ExchangeType.uniswap_v3,
                pair_count=1,
            ),
        }
    )

    # Create a test trading pair
    # Note: These addresses need to be updated with actual pool addresses on Arbitrum Sepolia
    weth_usdc_uniswap_v3 = TradingPairIdentifier(
        base=weth,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000000",  # TODO: Add actual pool address
        exchange_address="0x0000000000000000000000000000000000000000",  # TODO: Add actual factory address
        fee=0.0005,
    )

    universe = create_universe_from_trading_pair_identifiers(
        [weth_usdc_uniswap_v3],
        exchange_universe=exchange_universe,
    )
    return universe


@pytest.fixture(scope='module')
def orderly_strategy_universe(
    orderly_pair_universe,
) -> TradingStrategyUniverse:
    """Trading strategy universe for Orderly testing"""

    universe = Universe(
        chains={ChainId.arbitrum_sepolia},
        time_bucket=TimeBucket.not_applicable,
        exchange_universe=orderly_pair_universe.exchange_universe,
        pairs=orderly_pair_universe,
    )

    usdc = orderly_pair_universe.get_token("0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d")
    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets={translate_token(usdc)},
    )


@pytest.fixture
def vault_pair(weth, usdc, orderly_vault_address) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        base=weth,  # Vault shares, not used
        quote=usdc,  # USDC denomination
        pool_address=orderly_vault_address,
        exchange_address="0x0000000000000000000000000000000000000000",
        fee=0.0,
        kind=TradingPairKind.vault,
    )
