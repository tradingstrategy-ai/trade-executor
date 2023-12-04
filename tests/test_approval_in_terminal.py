"""Test manual approval of trades from CLI user interface."""

import logging
import os
import datetime
import secrets
from pathlib import Path
from typing import List

import pytest
from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.token import create_token
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, deploy_trading_pair, deploy_uniswap_v2_like
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.ethereum.hot_wallet_sync_model import EthereumHotWalletReserveSyncer, HotWalletSyncModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution_v0 import UniswapV2ExecutionModelVersion0
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier

from tradeexecutor.strategy.bootstrap import import_strategy_file
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.qstrader import HAS_QSTRADER
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StaticUniverseModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


pytestmark = pytest.mark.skipif(HAS_QSTRADER is False, reason="Install with optional qstrader dependency to run these tests")


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture
def tester_provider():
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return EthereumTesterProvider()


@pytest.fixture
def eth_tester(tester_provider):
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return tester_provider.ethereum_tester


@pytest.fixture
def web3(tester_provider):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(tester_provider)


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deploy account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def hot_wallet_private_key(web3) -> HexBytes:
    """Generate a private key"""
    return HexBytes(secrets.token_bytes(32))


@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Create USDC with 10M supply."""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 10_000_000 * 10**6, 6)
    return token


@pytest.fixture()
def uniswap_v2(web3, deployer) -> UniswapV2Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v2_like(web3, deployer)
    return deployment


@pytest.fixture
def weth_token(uniswap_v2: UniswapV2Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v2.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """USDC in trade executor domain representation."""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """WETH in trade executor domain representation."""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v2, weth_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 1.7M liquidity."""
    pair_address = deploy_trading_pair(
        web3,
        deployer,
        uniswap_v2,
        weth_token,
        usdc_token,
        1000 * 10**18,  # 1000 ETH liquidity
        1_700_000 * 10**6,  # 1.7M USDC liquidity
    )
    return pair_address


@pytest.fixture
def weth_usdc_pair(uniswap_v2, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    """WETH-USDC pair representation in the trade executor domain."""
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        weth_usdc_uniswap_trading_pair,
        uniswap_v2.factory.address,
        int(weth_usdc_uniswap_trading_pair, 16),
        fee=0.0030,
    )


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """What reserve currencies we support for the strategy."""
    return [usdc]


@pytest.fixture()
def hot_wallet(web3: Web3, usdc_token: Contract, hot_wallet_private_key: HexBytes, deployer: HexAddress) -> HotWallet:
    """Our trading Ethereum account.

    Start with 10,000 USDC cash and 2 ETH.
    """
    account = Account.from_key(hot_wallet_private_key)
    web3.eth.send_transaction({"from": deployer, "to": account.address, "value": 2*10**18})
    usdc_token.functions.transfer(account.address, 10_000 * 10**6).transact({"from": deployer})
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture
def supported_reserves(asset_usdc) -> List[AssetIdentifier]:
    """The reserve currencies we support."""
    return [asset_usdc]


@pytest.fixture()
def portfolio() -> Portfolio:
    """A portfolio loaded with the initial cash.

    We start with 10,000 USDC.
    """
    portfolio = Portfolio()
    return portfolio


@pytest.fixture()
def state(portfolio) -> State:
    return State(portfolio=portfolio)


@pytest.fixture()
def exchange_universe(web3, uniswap_v2: UniswapV2Deployment) -> ExchangeUniverse:
    """We trade on one Uniswap v2 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v2])


@pytest.fixture()
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_pair) -> PandasPairUniverse:
    """We trade on two trading pairs."""
    exchange = next(iter(exchange_universe.exchanges.values()))  # Get the first exchange from the universe
    return create_pair_universe(web3, exchange, [weth_usdc_pair])


@pytest.fixture()
def universe(web3, exchange_universe: ExchangeUniverse, pair_universe: PandasPairUniverse) -> Universe:
    """Get our trading universe."""
    return Universe(
        time_bucket=TimeBucket.d1,
        chains=[ChainId(web3.eth.chain_id)],
        exchanges=set(exchange_universe.exchanges.values()),
        pairs=pair_universe,
        candles=GroupedCandleUniverse.create_empty_qstrader(),
        liquidity=GroupedLiquidityUniverse.create_empty(),
    )


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../strategies/test_only", "random_eth_usdc.py"))


@pytest.fixture()
def recorded_input() -> bool:
    """Do we do interactive execution where the user presses the key, or use recorded key presses."""
    return os.environ.get("USER_INTERACTION") is None


@pytest.fixture()
def routing_model(uniswap_v2, asset_usdc, asset_weth, weth_usdc_pair) -> UniswapV2Routing:

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        uniswap_v2.factory.address: (uniswap_v2.router.address, uniswap_v2.init_code_hash),
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        asset_weth.address: weth_usdc_pair.pool_address
    }

    return UniswapV2Routing(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=asset_usdc.address,
        trading_fee=0.0,
    )


def test_cli_approve_trades(
        logger: logging.Logger,
        strategy_path: Path,
        web3: Web3,
        hot_wallet: HotWallet,
        uniswap_v2: UniswapV2Deployment,
        universe,
        routing_model,
        state: State,
        supported_reserves,
        weth_usdc_pair,
        weth_token,
        usdc_token,
        recorded_input,
    ):
    """CLI approval dialog for choosing which new trades to approve."""

    factory = import_strategy_file(strategy_path)
    approval_model = CLIApprovalModel()
    execution_model = UniswapV2ExecutionModelVersion0(uniswap_v2, hot_wallet, confirmation_block_count=0)
    # sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    valuation_model_factory = uniswap_v2_sell_valuation_factory
    pricing_model_factory = uniswap_v2_live_pricing_factory
    executor_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=supported_reserves)
    universe_model = StaticUniverseModel(executor_universe)

    description: StrategyExecutionDescription = factory(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=approval_model,
        valuation_model_factory=valuation_model_factory,
        sync_model=sync_model,
        client=None,
        pricing_model_factory=pricing_model_factory,
        routing_model=routing_model,
        universe_model=universe_model)

    runner = description.runner

    debug_details = {}
    if recorded_input:
        # See hints at https://github.com/MarcoMernberger/mgenomicsremotemail/blob/ac5fbeaf02ae80b0c573c6361c9279c540b933e4/tests/tmp.py#L27
        with create_pipe_input() as inp:
            keys = " \t\r"  # Toggle checkbox with space, tab to ok, press enter
            inp.send_text(keys)
            with create_app_session(input=inp, output=DummyOutput()):
                runner.tick(datetime.datetime(2020, 1, 1), executor_universe, state, debug_details)
    else:
        runner.tick(datetime.datetime(2020, 1, 1), executor_universe, state, debug_details)

    assert len(debug_details["alpha_model_weights"]) == 1
    assert len(debug_details["approved_trades"]) == 1


def test_cli_disapprove_trades(
        logger: logging.Logger,
        strategy_path: Path,
        web3: Web3,
        hot_wallet: HotWallet,
        uniswap_v2: UniswapV2Deployment,
        universe,
        state: State,
        routing_model,
        supported_reserves,
        weth_usdc_pair,
        weth_token,
        usdc_token,
        recorded_input,
    ):
    """CLI approval dialog can approve trades."""

    factory = import_strategy_file(strategy_path)
    approval_model = CLIApprovalModel()
    execution_model = UniswapV2ExecutionModelVersion0(uniswap_v2, hot_wallet, confirmation_block_count=0)
    # sync_method = EthereumHotWalletReserveSyncer(web3, hot_wallet.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)
    valuation_model_factory = uniswap_v2_sell_valuation_factory
    pricing_model_factory = uniswap_v2_live_pricing_factory
    executor_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=supported_reserves)
    universe_model = StaticUniverseModel(executor_universe)

    description: StrategyExecutionDescription = factory(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=approval_model,
        valuation_model_factory=valuation_model_factory,
        sync_model=sync_model,
        routing_model=routing_model,
        client=None,
        pricing_model_factory=pricing_model_factory,
        universe_model=universe_model)

    runner = description.runner

    debug_details = {}
    if recorded_input:
        # See hints at https://github.com/MarcoMernberger/mgenomicsremotemail/blob/ac5fbeaf02ae80b0c573c6361c9279c540b933e4/tests/tmp.py#L27
        with create_pipe_input() as inp:
            keys = "\t\r"  # Skip checkbox with tab, press enter
            inp.send_text(keys)
            with create_app_session(input=inp, output=DummyOutput()):
                runner.tick(datetime.datetime(2020, 1, 1), executor_universe, state, debug_details)
    else:
        runner.tick(datetime.datetime(2020, 1, 1), executor_universe, state, debug_details)

    assert len(debug_details["alpha_model_weights"]) == 1
    assert len(debug_details["approved_trades"]) == 0
