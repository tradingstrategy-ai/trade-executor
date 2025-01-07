"""Base mainnet fork based tests for Lagoon.

Explore the static deployment which we fork from the Base mainnet:

- Vault UI: https://trading-stategy-users-frontend.vercel.app/vault/8453/0xab4ac28d10a4bc279ad073b1d74bfa0e385c010c
- Vault contract: https://basescan.org/address/0xab4ac28d10a4bc279ad073b1d74bfa0e385c010c#readProxyContract
- Safe UI: https://app.safe.global/home?safe=base:0x20415f3Ec0FEA974548184bdD6e67575D128953F
- Safe contract: https://basescan.org/address/0x20415f3Ec0FEA974548184bdD6e67575D128953F#readProxyContract
- Roles: https://app.safe.global/apps/open?safe=base:0x20415f3Ec0FEA974548184bdD6e67575D128953F&appUrl=https%3A%2F%2Fzodiac.gnosisguild.org%2F
"""
import os
from decimal import Decimal

import pytest
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.hotwallet import HotWallet
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
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
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
def anvil_base_fork(request, usdc_holder) -> AnvilLaunch:
    """Create a testable fork of live BNB chain.

    :return: JSON-RPC URL for Web3
    """
    assert JSON_RPC_BASE, "JSON_RPC_BASE not set"
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        unlocked_addresses=[usdc_holder],
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
def base_doginme() -> AssetIdentifier:
    """DogInMe"""
    return AssetIdentifier(
        chain_id=8453,
        address="0x6921B130D297cc43754afba22e5EAc0FBf8Db75b",
        decimals=18,
        token_symbol="DogInMe",
    )


@pytest.fixture(scope='module')
def base_ski() -> AssetIdentifier:
    """SKI - SkiMask.

    https://app.uniswap.org/explore/tokens/base/0x768be13e1680b5ebe0024c42c896e3db59ec0149

    .. warning::

        Has weird tax "cooling off period"

        https://basescan.org/address/0x768BE13e1680b5ebE0024C42c896E3dB59ec0149#code


    """
    return AssetIdentifier(
        chain_id=8453,
        address="0x768BE13e1680b5ebE0024C42c896E3dB59ec0149",
        decimals=18,
        token_symbol="SKI",
    )


@pytest.fixture(scope='module')
def base_keycat() -> AssetIdentifier:
    """KEYCAT

    - Uniswap v2

    https://dexscreener.com/base/0x377feeed4820b3b28d1ab429509e7a0789824fca
    """
    return AssetIdentifier(
        chain_id=8453,
        address="0x9a26F5433671751C3276a065f57e5a02D2817973",
        decimals=18,
        token_symbol="KEYCAT",
    )


@pytest.fixture()
def base_doginme_token(web3, base_doginme) -> TokenDetails:
    """DogInMe.

    - Uniswap v3

    https://app.uniswap.org/explore/tokens/base/0x6921b130d297cc43754afba22e5eac0fbf8db75b
    """
    return fetch_erc20_details(web3, base_doginme.address)


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
def hot_wallet(web3, usdc, usdc_holder) -> HotWallet:
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

    # Top up with 999 USDC
    tx_hash = usdc.contract.functions.transfer(hw.address, 999 * 10**6).transact({"from": usdc_holder, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return hw


@pytest.fixture()
def base_test_vault_spec() -> VaultSpec:
    """Vault is 0xab4ac28d10a4bc279ad073b1d74bfa0e385c010c

    - https://trading-stategy-users-frontend.vercel.app/vault/8453/0xab4ac28d10a4bc279ad073b1d74bfa0e385c010c
    - https://app.safe.global/home?safe=base:0x20415f3Ec0FEA974548184bdD6e67575D128953F
    """
    return VaultSpec(1, "0xab4ac28d10a4bc279ad073b1d74bfa0e385c010c")


@pytest.fixture()
def lagoon_vault(web3, base_test_vault_spec: VaultSpec) -> LagoonVault:
    return LagoonVault(web3, base_test_vault_spec)


@pytest.fixture()
def asset_manager(web3) -> HotWallet:
    """Account that we use for Lagoon trade/settles"""
    hot_wallet = HotWallet.create_for_testing(web3, eth_amount=1)
    return hot_wallet


@pytest.fixture()
def topped_up_valuation_manager(web3, valuation_manager):
    # Topped up with some ETH
    tx_hash = web3.eth.send_transaction({
        "to": valuation_manager,
        "from": web3.eth.accounts[0],
        "value": 9 * 10**18,
    })
    assert_transaction_success_with_explanation(web3, tx_hash)
    return valuation_manager



@pytest.fixture()
def lagoon_execution_model(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    asset_manager: HotWallet,
) -> LagoonExecution:
    """Set EthereumExecutionModel in Base fork testing mode."""

    vault = automated_lagoon_vault.vault

    execution_model = LagoonExecution(
        vault=vault,
        tx_builder=LagoonTransactionBuilder(vault, asset_manager),
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    return execution_model


@pytest.fixture()
def lagoon_routing_model(lagoon_execution_model, vault_strategy_universe) -> GenericRouting:
    return lagoon_execution_model.create_default_routing_model(vault_strategy_universe)


@pytest.fixture()
def lagoon_pricing_model(
    web3,
    vault_strategy_universe,
) -> GenericPricing:
    pair_configurator = EthereumPairConfigurator(
        web3,
        vault_strategy_universe,
    )

    weth_usdc = vault_strategy_universe.get_pair_by_human_description(
        (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
    )

    base_weth = weth_usdc.base

    return GenericPricing(
        pair_configurator,
        exchange_rate_pairs={
            base_weth: weth_usdc,
        }
    )


@pytest.fixture(scope='module')
def vault_pair_universe(
    base_usdc,
    base_doginme,
    base_weth,
    base_ski,
    base_keycat,
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
            ),

            2: Exchange(
                chain_id=ChainId(8453),
                chain_slug="base",
                exchange_id=2,
                exchange_slug="uniswap-v2",
                address="0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",
                exchange_type=ExchangeType.uniswap_v2,
                pair_count=1,
            )
        }
    )

    trading_pair_uniswap_v3 = TradingPairIdentifier(
        base=base_doginme,
        quote=base_weth,
        pool_address="0xADE9BcD4b968EE26Bed102dd43A55f6A8c2416df",
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        fee=0.0100,
    )

    trading_pair_uniswap_v2 = TradingPairIdentifier(
        base=base_ski,
        quote=base_weth,
        pool_address="0x6d6391b9bd02eefa00fa711fb1cb828a6471d283",
        exchange_address="0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",  # Uniswap v2 factory on Base
        fee=0.0030,
    )

    trading_pair_2_uniswap_v2 = TradingPairIdentifier(
        base=base_keycat,
        quote=base_weth,
        pool_address="0x377FeeeD4820B3B28D1ab429509e7A0789824fCA",
        exchange_address="0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",  # Uniswap v2 factory on Base
        fee=0.0030,
    )

    # https://coinmarketcap.com/dexscan/base/0xd0b53d9277642d899df5c87a3966a349a798f224/
    weth_usdc_uniswap_v3 = TradingPairIdentifier(
        base=base_weth,
        quote=base_usdc,
        pool_address="0xd0b53d9277642d899df5c87a3966a349a798f224",
        exchange_address="0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        fee=0.0005,
    )

    # https://app.uniswap.org/explore/pools/base/0x88A43bbDF9D098eEC7bCEda4e2494615dfD9bB9C
    weth_usdc_uniswap_v2 = TradingPairIdentifier(
        base=base_weth,
        quote=base_usdc,
        pool_address="0x88A43bbDF9D098eEC7bCEda4e2494615dfD9bB9C",
        exchange_address="0x8909Dc15e40173Ff4699343b6eB8132c65e18eC6",
        fee=0.0030,
    )

    universe = create_universe_from_trading_pair_identifiers(
        [
            trading_pair_uniswap_v3,
            trading_pair_uniswap_v2,
            weth_usdc_uniswap_v3,
            weth_usdc_uniswap_v2,
            trading_pair_2_uniswap_v2
        ],
        exchange_universe=exchange_universe,
    )
    return universe


@pytest.fixture(scope='module')
def vault_strategy_universe(
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
def automated_lagoon_vault(
    web3,
    deployer_local_account,
    asset_manager,
    multisig_owners,
    uniswap_v2,
    uniswap_v3,
) -> LagoonAutomatedDeployment:
    """Deploy a new Lagoon vault with TradingStrategyModuleV0.

    - Whitelist any Uniswap v2 token for trading using TradingStrategyModuleV0 and asset_manager
    """

    chain_id = web3.eth.chain_id
    deployer = deployer_local_account

    parameters = LagoonDeploymentParameters(
        underlying=USDC_NATIVE_TOKEN[chain_id],
        name="Example",
        symbol="EXA",
    )

    deploy_info = deploy_automated_lagoon_vault(
        web3=web3,
        deployer=deployer,
        asset_manager=asset_manager.address,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=2,
        uniswap_v2=uniswap_v2,
        uniswap_v3=uniswap_v3,
        any_asset=True,
    )

    return deploy_info


@pytest.fixture()
def deposited_lagoon_vault(
    web3,
    automated_lagoon_vault,
    depositor,
    base_usdc_token,
):
    """Lagoon vault with deposits in it.

    - Does the initial Vault.deposit() Solidity call
    - Treasury not yet synced
    """

    vault = automated_lagoon_vault.vault
    usdc = base_usdc_token

    # Do initial deposit
    # Deposit 399.00 USDC into the vault from the first user
    usdc_amount = Decimal(399.00)
    raw_usdc_amount = usdc.convert_to_raw(usdc_amount)
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, raw_usdc_amount)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return automated_lagoon_vault


@pytest.fixture()
def depositor(web3, base_usdc_token, usdc_holder) -> HexAddress:
    """User with some USDC ready to deposit.

    - Start with 500 USDC
    """
    new_depositor = web3.eth.accounts[5]
    tx_hash = base_usdc_token.transfer(new_depositor, Decimal(500)).transact({"from": usdc_holder, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return new_depositor


@pytest.fixture()
def another_new_depositor(web3, base_usdc_token, usdc_holder) -> HexAddress:
    """User with some USDC ready to deposit.

    - Start with 500 USDC
    - We need two test users
    """
    another_new_depositor = web3.eth.accounts[6]
    tx_hash = base_usdc_token.transfer(another_new_depositor, Decimal(500)).transact({"from": usdc_holder, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return another_new_depositor


@pytest.fixture()
def uniswap_v2(web3) -> UniswapV2Deployment:
    """Uniswap V2 on Base"""
    return fetch_deployment(
        web3,
        factory_address=UNISWAP_V2_DEPLOYMENTS["base"]["factory"],
        router_address=UNISWAP_V2_DEPLOYMENTS["base"]["router"],
        init_code_hash=UNISWAP_V2_DEPLOYMENTS["base"]["init_code_hash"],
    )


@pytest.fixture()
def uniswap_v3(web3) -> UniswapV3Deployment:
    deployment_data = UNISWAP_V3_DEPLOYMENTS["base"]
    uniswap_v3_on_base = fetch_deployment_uni_v3(
        web3,
        factory_address=deployment_data["factory"],
        router_address=deployment_data["router"],
        position_manager_address=deployment_data["position_manager"],
        quoter_address=deployment_data["quoter"],
        quoter_v2=deployment_data["quoter_v2"],
        router_v2=deployment_data["router_v2"],
    )
    return uniswap_v3_on_base


@pytest.fixture()
def deployer_local_account(web3) -> LocalAccount:
    """Account that we use for Lagoon deployment"""
    hot_wallet = HotWallet.create_for_testing(web3, eth_amount=1)
    return hot_wallet.account


@pytest.fixture()
def multisig_owners(web3) -> list[HexAddress]:
    """Accouunts that are set as the owners of deployed Safe w/valt"""
    return [web3.eth.accounts[2], web3.eth.accounts[3], web3.eth.accounts[4]]
