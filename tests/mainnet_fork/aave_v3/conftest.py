"""Fixtures to set up Ethereum mainnet fork generic router for uniswap v3 + aave v3"""

import os
import pandas as pd
import pytest as pytest
from web3 import Web3

from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import HexAddress, HexStr

from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.provider.anvil import fork_network_anvil, mine
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.provider.multi_provider import create_multi_provider_web3

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.ethereum.universe import create_exchange_universe
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options
from tradeexecutor.state.identifier import AssetIdentifier




@pytest.fixture(scope="module")
def large_usdc_holder() -> HexAddress:
    """A random account picked from Ethereum that holds a lot of USDC.

    This account is unlocked on Anvil, so you have access to good USDC stash.

    `To find large holder accounts, use <https://etherscan.io/token/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48#balances>`_.
    """
    # Binance Hot Wallet
    return HexAddress(HexStr("0x5041ed759Dd4aFc3a72b8192C143F72f4724081A"))


@pytest.fixture
def anvil_ethereum_fork(request, large_usdc_holder) -> str:
    """Create a testable fork of live Arbitrum.

    :return: JSON-RPC URL for Web3
    """
    launch = fork_network_anvil(
        os.environ["JSON_RPC_ETHEREUM"],
        unlocked_addresses=[large_usdc_holder],
        fork_block_number=20_000_000,
    )
    try:
        yield launch.json_rpc_url
    finally:
        # Wind down Anvil process after the test is complete
        # launch.close(log_level=logging.ERROR)
        launch.close()


@pytest.fixture
def web3(anvil_ethereum_fork: str):
    """Set up a Web3 provider instance with a lot of workarounds for flaky nodes."""
    return create_multi_provider_web3(anvil_ethereum_fork)


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
    """Hot wallet used for fork test."""
    assert isinstance(user_1, LocalAccount)
    wallet = HotWallet(user_1)
    wallet.sync_nonce(web3)

    # give hot wallet some native token and USDC
    web3.eth.send_transaction(
        {
            "from": large_usdc_holder,
            "to": wallet.address,
            "value": 5 * 10**18,
        }
    )

    usdc.contract.functions.transfer(
        wallet.address,
        10_000 * 10**6,
    ).transact({"from": large_usdc_holder})

    # print(usdc.contract.functions.balanceOf(wallet.address).call())

    wallet.sync_nonce(web3)

    # mine a few blocks
    for i in range(1, 10):
        mine(web3)

    return wallet


@pytest.fixture()
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


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
def usdc(web3) -> TokenDetails:
    """Get USDC on Ethereum."""
    return fetch_erc20_details(web3, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")


@pytest.fixture
def asset_usdc(usdc, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(
        chain_id,
        usdc.contract.address,
        usdc.symbol,
        usdc.decimals,
    )


@pytest.fixture()
def strategy_universe(
    chain_id,
    exchange_universe,
    asset_usdc,
    persistent_test_client
) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""

    pairs = [
        (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.ethereum, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.ethereum, LendingProtocolType.aave_v3, "USDC"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2024-05-25"),
        end_at=pd.Timestamp("2024-06-10"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_from_dataset(dataset, asset_usdc.address)


@pytest.fixture()
def pair_configurator(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
) -> EthereumPairConfigurator:
    return EthereumPairConfigurator(
        web3,
        strategy_universe,
    )


@pytest.fixture()
def generic_routing_model(pair_configurator) -> GenericRouting:
    return GenericRouting(pair_configurator)


@pytest.fixture()
def generic_pricing_model(pair_configurator) -> GenericPricing:
    return GenericPricing(pair_configurator)


@pytest.fixture()
def generic_valuation_model(pair_configurator) -> GenericValuation:
    return GenericValuation(pair_configurator)


# @pytest.fixture
# def state(web3, hot_wallet, asset_usdc) -> State:
#     """State used in the tests."""
#     state = State()

#     events = sync_reserves(
#         web3, datetime.datetime.utcnow(), hot_wallet.address, [], [asset_usdc]
#     )
#     assert len(events) > 0
#     apply_sync_events(state, events)
#     reserve_currency, exchange_rate = state.portfolio.get_default_reserve_asset()
#     assert reserve_currency == asset_usdc
#     return state
