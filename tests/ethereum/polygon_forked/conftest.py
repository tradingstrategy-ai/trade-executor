"""Set up assets and contracts for Aave and 1delta on Polygon fork."""

import os
import logging

import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress, HexStr

from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.hotwallet import HotWallet
from eth_defi.aave_v3.deployment import AaveV3Deployment, fetch_deployment as fetch_aave_deployment
from eth_defi.one_delta.deployment import OneDeltaDeployment
from eth_defi.one_delta.deployment import fetch_deployment as fetch_1delta_deployment
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.anvil import fork_network_anvil, mine
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.ethereum.routing_data import get_quickswap_default_routing_parameters
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind, AssetType
from tradeexecutor.strategy.reserve_currency import ReserveCurrency

WETH_USDC_FEE = 0.003
AAVE_USDC_FEE = 0.003

WETH_USDC_FEE_RAW = 3000
AAVE_USDC_FEE_RAW = 3000


@pytest.fixture(scope="module")
def large_usdc_holder() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC.

    This account is unlocked on Anvil, so you have access to good USDC stash.

    `To find large holder accounts, use <https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances>`_.
    """
    # Binance Hot Wallet 6
    return HexAddress(HexStr("0xe7804c37c13166fF0b37F5aE0BB07A3aEbb6e245"))


@pytest.fixture
def anvil_polygon_chain_fork(request, large_usdc_holder) -> str:
    """Create a testable fork of live Polygon.

    :return: JSON-RPC URL for Web3
    """
    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]
    launch = fork_network_anvil(
        mainnet_rpc,
        unlocked_addresses=[large_usdc_holder],
        fork_block_number=51_000_000,
    )
    try:
        yield launch.json_rpc_url
    finally:
        # Wind down Anvil process after the test is complete
        # launch.close(log_level=logging.ERROR)
        launch.close()


@pytest.fixture
def web3(anvil_polygon_chain_fork: str):
    """Set up a Web3 provider instance with a lot of workarounds for flaky nodes."""
    return create_multi_provider_web3(anvil_polygon_chain_fork)


@pytest.fixture
def chain_id(web3) -> int:
    """The fork chain id."""
    return web3.eth.chain_id


@pytest.fixture(scope="module")
def user_1() -> LocalAccount:
    """Create a test account."""
    return Account.create()


@pytest.fixture
def hot_wallet(web3, user_1, usdc, large_usdc_holder) -> HotWallet:
    """Hot wallet used for fork tets.

    - Starts with MATIC and $10k USDC balance
    """
    assert isinstance(user_1, LocalAccount)
    wallet = HotWallet(user_1)
    wallet.sync_nonce(web3)

    # give hot wallet some native token and USDC
    web3.eth.send_transaction(
        {
            "from": large_usdc_holder,
            "to": wallet.address,
            "value": 100 * 10**18,
        }
    )

    usdc.contract.functions.transfer(
        wallet.address,
        10_000 * 10**6,
    ).transact({"from": large_usdc_holder})

    wallet.sync_nonce(web3)

    # mine a few blocks
    for i in range(1, 5):
        mine(web3)

    return wallet


@pytest.fixture
def quickswap_deployment(web3) -> UniswapV2Deployment:
    """Quickswap deployment on Polygon."""
    return fetch_deployment(
        web3,
        "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
        "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
        "0x96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f",
    )


@pytest.fixture
def uniswap_v3_deployment(web3) -> UniswapV3Deployment:
    """Uniswap v3 deployment."""
    return fetch_uniswap_v3_deployment(
        web3,
        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
    )


@pytest.fixture
def aave_v3_deployment(web3):
    return fetch_aave_deployment(
        web3,
        pool_address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        data_provider_address="0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654",
        oracle_address="0xb023e699F5a33916Ea823A16485e259257cA8Bd1",
    )


@pytest.fixture
def one_delta_deployment(web3) -> OneDeltaDeployment:
    return fetch_1delta_deployment(
        web3,
        flash_aggregator_address="0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
        broker_proxy_address="0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
        quoter_address="0x36de3876ad1ef477e8f6d98EE9a162926f00463A",
    )


@pytest.fixture
def usdc(web3) -> TokenDetails:
    """Get USDC on Polygon."""
    return fetch_erc20_details(web3, "0x2791bca1f2de4661ed88a30c99a7a9449aa84174")


@pytest.fixture
def ausdc(web3) -> TokenDetails:
    """Get aPolUSDC on Polygon."""
    return fetch_erc20_details(web3, "0x625E7708f30cA75bfd92586e17077590C60eb4cD", contract_name="aave_v3/AToken.json")


@pytest.fixture
def weth(web3) -> TokenDetails:
    """Get WETH on Polygon."""
    return fetch_erc20_details(web3, "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619")


@pytest.fixture
def vweth(web3) -> TokenDetails:
    """Get vPolWETH on Polygon."""
    return fetch_erc20_details(web3, "0x0c84331e39d6658Cd6e6b9ba04736cC4c4734351", contract_name="aave_v3/VariableDebtToken.json")


@pytest.fixture
def wmatic(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270")
    return details


@pytest.fixture
def price_helper(uniswap_v3_deployment):
    return UniswapV3PriceHelper(uniswap_v3_deployment)


@pytest.fixture
def asset_usdc(usdc, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        usdc.contract.address,
        usdc.symbol,
        usdc.decimals,
    )


@pytest.fixture
def asset_weth(weth, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        weth.contract.address,
        weth.symbol,
        weth.decimals,
    )


@pytest.fixture()
def asset_ausdc(ausdc, asset_usdc, chain_id) -> AssetIdentifier:
    """USDC collateral token."""
    return AssetIdentifier(
        chain_id,
        ausdc.contract.address,
        ausdc.symbol,
        ausdc.decimals,
        underlying=asset_usdc,
        type=AssetType.collateral,
        liquidation_threshold=0.85,  # From Aave UI
    )


@pytest.fixture
def asset_vweth(vweth, asset_weth, chain_id) -> AssetIdentifier:
    """Variable debt token."""
    return AssetIdentifier(
        chain_id,
        vweth.contract.address,
        vweth.symbol,
        vweth.decimals,
        underlying=asset_weth,
        type=AssetType.borrowed,
    )


@pytest.fixture()
def asset_wmatic(wmatic: TokenDetails) -> AssetIdentifier:
    """WETH as a persistent id.
    """
    return translate_token_details(wmatic)


@pytest.fixture
def weth_usdc_spot_pair(uniswap_v3_deployment, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        "0x0e44ceb592acfc5d3f09d996302eb4c499ff8c10",
        uniswap_v3_deployment.factory.address,
        fee=WETH_USDC_FEE,
    )


@pytest.fixture
def wmatic_usdc_spot_pair(quickswap_deployment, asset_usdc, asset_wmatic) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_wmatic,
        asset_usdc,
        "0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827",
        quickswap_deployment.factory.address,
        fee=0.003,
    )


@pytest.fixture
def weth_usdc_shorting_pair(uniswap_v3_deployment, asset_ausdc, asset_vweth, weth_usdc_spot_pair) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_vweth,
        asset_ausdc,
        "0x1",  # TODO
        uniswap_v3_deployment.factory.address,
        # fee=WETH_USDC_FEE,
        kind=TradingPairKind.lending_protocol_short,
        underlying_spot_pair=weth_usdc_spot_pair,
    )


@pytest.fixture()
def one_delta_routing_model(
    one_delta_deployment,
    aave_v3_deployment,
    uniswap_v3_deployment,
    asset_usdc,
    asset_weth,
) -> OneDeltaRouting:
    return OneDeltaRouting(
        address_map={
            "one_delta_broker_proxy": one_delta_deployment.broker_proxy.address,
            "one_delta_quoter": one_delta_deployment.quoter.address,
            "aave_v3_pool": aave_v3_deployment.pool.address,
            "aave_v3_data_provider": aave_v3_deployment.data_provider.address,
            "aave_v3_oracle": aave_v3_deployment.oracle.address,
            "factory": uniswap_v3_deployment.factory.address,
            "router": uniswap_v3_deployment.swap_router.address,
            "position_manager": uniswap_v3_deployment.position_manager.address,
            "quoter": uniswap_v3_deployment.quoter.address
        },
        allowed_intermediary_pairs={},
        reserve_token_address=asset_usdc.address.lower(),
    )


# @pytest.fixture()
# def uniswap_v3_routing_model(asset_usdc) -> UniswapV3Routing:

#     # for uniswap v3
#     # same addresses for Mainnet, Polygon, Optimism, Arbitrum, Testnets Address
#     # only celo different
#     address_map = {
#         "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
#         "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
#         "position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
#         "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
#         # "router02":"0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
#         # "quoterV2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
#     }

#     allowed_intermediary_pairs = {
#         # Route WMATIC through USDC:WMATIC fee 0.05% pool,
#         # https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/matic-usdc-fee-5
#         "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0xa374094527e1673a86de625aa59517c5de346d32",
#         # Route WETH through USDC:WETH fee 0.05% pool,
#         # https://tradingstrategy.ai/trading-view/polygon/uniswap-v3/eth-usdc-fee-5
#         "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "0x45dda9cb7c25131df268515131f647d726f50608",
#     }

#     return UniswapV3Routing(
#         address_map,
#         allowed_intermediary_pairs,
#         reserve_token_address=asset_usdc.address,
#     )


# @pytest.fixture()
# def quickswap_routing_model(
#         quickswap_deployment,
#         wmatic_usdc_spot_pair,
#         asset_usdc,
# ) -> UniswapV2Routing:
#     # Route WMATIC and USDC quoted pairs on Quickswap
#     uniswap_v2_router = UniswapV2Routing(
#         factory_router_map={quickswap_deployment.factory.address: (quickswap_deployment.router.address, quickswap_deployment.init_code_hash)},
#         allowed_intermediary_pairs={
#             wmatic_usdc_spot_pair.base.address: wmatic_usdc_spot_pair.pool_address,
#             # "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "0x45dda9cb7c25131df268515131f647d726f50608",
#         },
#         reserve_token_address=asset_usdc.address,
#     )
#     return uniswap_v2_router
