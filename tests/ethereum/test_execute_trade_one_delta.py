"""Test trading and portfolio management against 1delta using forked Polygon environment."""

import os
import shutil
import datetime
from decimal import Decimal
from typing import List

import pytest
from eth_account import Account
from eth_typing import HexAddress, HexStr
from eth_account.signers.local import LocalAccount

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment as fetch_uniswap_v3_deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from eth_defi.aave_v3.deployment import AaveV3Deployment, fetch_deployment as fetch_aave_deployment
from eth_defi.one_delta.deployment import OneDeltaDeployment
from eth_defi.one_delta.deployment import fetch_deployment as fetch_1delta_deployment
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.anvil import fork_network_anvil, mine

from tradeexecutor.ethereum.universe import create_pair_universe
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.ethereum.balance_update import apply_reserve_update_events
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, AssetType, TradingPairKind
from tradeexecutor.testing.ethereumtrader_one_delta import OneDeltaTestTrader
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradeexecutor.ethereum.one_delta.analysis import decode_path


pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)

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
    """Hot wallet."""
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
def usdc(web3):
    """Get USDC on Polygon."""
    return fetch_erc20_details(web3, "0x2791bca1f2de4661ed88a30c99a7a9449aa84174")


@pytest.fixture
def ausdc(web3):
    """Get aPolUSDC on Polygon."""
    return fetch_erc20_details(web3, "0x625E7708f30cA75bfd92586e17077590C60eb4cD", contract_name="aave_v3/AToken.json")


@pytest.fixture
def weth(web3):
    """Get WETH on Polygon."""
    return fetch_erc20_details(web3, "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619")


@pytest.fixture
def vweth(web3):
    """Get vPolWETH on Polygon."""
    return fetch_erc20_details(web3, "0x0c84331e39d6658Cd6e6b9ba04736cC4c4734351", contract_name="aave_v3/VariableDebtToken.json")


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


@pytest.fixture
def weth_usdc_spot_pair(uniswap_v3_deployment, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        "0x1",  # TODO
        uniswap_v3_deployment.factory.address,
        fee=WETH_USDC_FEE,
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


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(asset_usdc) -> List[AssetIdentifier]:
    """The reserve currencies we support."""
    return [asset_usdc]


@pytest.fixture
def state(web3, hot_wallet, asset_usdc, usdc) -> State:
    """State used in the tests."""
    state = State()

    events = sync_reserves(
        web3, datetime.datetime.utcnow(), hot_wallet.address, [], [asset_usdc]
    )
    assert len(events) > 0
    apply_reserve_update_events(state, events)
    reserve_currency, exchange_rate = state.portfolio.get_default_reserve_asset()
    assert reserve_currency == asset_usdc
    return state


@pytest.fixture
def pair_universe(web3, weth_usdc_spot_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_spot_pair])


@pytest.fixture
def tx_builder(web3, hot_wallet) -> HotWalletTransactionBuilder:
    tx_builder = HotWalletTransactionBuilder(
        web3,
        hot_wallet,
    )
    return tx_builder


@pytest.fixture
def ethereum_trader(
    web3: Web3,
    uniswap_v3_deployment: UniswapV3Deployment,
    aave_v3_deployment: AaveV3Deployment,
    one_delta_deployment: OneDeltaDeployment,
    hot_wallet: HotWallet,
    state: State,
    pair_universe: PandasPairUniverse,
    tx_builder: HotWalletTransactionBuilder,
) -> OneDeltaTestTrader:
    return OneDeltaTestTrader(
        one_delta_deployment,
        aave_v3_deployment,
        uniswap_v3_deployment,
        state,
        pair_universe,
        tx_builder,
    )


def test_one_delta_decode_path():
    encoded = bytes.fromhex("7ceb23fd6bc0add59e62ac25578270cff1b9f619000bb800062791bca1f2de4661ed88a30c99a7a9449aa8417402")
    decoded = decode_path(encoded)

    assert decoded == ['0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619', 3000, 0, 6, '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174']


def test_execute_trade_instructions_open_short(
    web3: Web3,
    state: State,
    pair_universe: PandasPairUniverse,
    uniswap_v3_deployment: UniswapV3Deployment,
    hot_wallet: HotWallet,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
    weth_usdc_shorting_pair: TradingPairIdentifier,
    start_ts: datetime.datetime,
    price_helper: UniswapV3PriceHelper,
    ethereum_trader: OneDeltaTestTrader 
):
    """Open short position."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    # Buy 500 USDC worth of WETH
    trader = UnitTestTrader(state)

    # swap from quote to base (usdc to weth)
    path = [usdc.contract.address, weth.contract.address]
    fees = [WETH_USDC_FEE_RAW]
    eth_price = price_helper.get_amount_in(1 * 10 ** 18, path, fees) / 10 ** 6
    
    assert eth_price == pytest.approx(2250.367818)

    reserve_amount = Decimal(5000)
    leverage = 2

    position, trade = trader.open_short(
        weth_usdc_shorting_pair,
        reserve_amount,
        eth_price,
        leverage,
    )
    assert trade.is_leverage()
    assert state.portfolio.calculate_total_equity() == pytest.approx(10000.0)
    assert trade.get_status() == TradeStatus.planned

    ethereum_trader.execute_trades_simple(ethereum_trader.create_routing_model(), [trade])

    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(2225.901440877395)
    assert abs(trade.executed_quantity) == pytest.approx(Decimal(4.443718009124141875))
    # TODO:
    assert trade.lp_fees_paid == pytest.approx(0.013331154027372427)
    assert trade.native_token_price == 0.0

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1
    position = portfolio.open_positions[1]
    assert position.get_collateral() == pytest.approx(14970)
    assert position.get_borrowed() == pytest.approx(10000)
    assert position.get_value() == pytest.approx(4970)


def test_execute_trade_instructions_open_and_close_short(
    web3: Web3,
    state: State,
    pair_universe: PandasPairUniverse,
    uniswap_v3_deployment: UniswapV3Deployment,
    hot_wallet: HotWallet,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
    weth_usdc_shorting_pair: TradingPairIdentifier,
    start_ts: datetime.datetime,
    price_helper: UniswapV3PriceHelper,
    ethereum_trader: OneDeltaTestTrader 
):
    """Open short position."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    # Buy 500 USDC worth of WETH
    trader = UnitTestTrader(state)

    # swap from quote to base (usdc to weth)
    path = [usdc.contract.address, weth.contract.address]
    fees = [WETH_USDC_FEE_RAW]
    eth_price = price_helper.get_amount_in(1 * 10 ** 18, path, fees) / 10 ** 6
    
    assert eth_price == pytest.approx(2250.367818)

    reserve_amount = Decimal(5000)
    leverage = 2

    position1, trade1 = trader.open_short(
        weth_usdc_shorting_pair,
        reserve_amount,
        eth_price,
        leverage,
    )
    assert trade1.is_leverage()
    assert state.portfolio.calculate_total_equity() == pytest.approx(10000.0)
    assert trade1.get_status() == TradeStatus.planned

    ethereum_trader.execute_trades_simple(ethereum_trader.create_routing_model(), [trade1])

    assert len(state.portfolio.open_positions) == 1
    assert trade1.get_status() == TradeStatus.success
    assert trade1.executed_price == pytest.approx(2225.901440877395)
    assert abs(trade1.executed_quantity) == pytest.approx(Decimal(4.443718009124141875))
    assert trade1.native_token_price == 0.0

    position2, trade2 = trader.close_short(
        weth_usdc_shorting_pair,
        reserve_amount, # TODO
        eth_price,
        leverage,
    )

    assert trade2.is_leverage()
    assert trade2.get_status() == TradeStatus.planned

    ethereum_trader.execute_trades_simple(ethereum_trader.create_routing_model(), [trade2])

    assert trade2.get_status() == TradeStatus.success
    assert trade2.executed_price == pytest.approx(2241.9849582746133)
    assert abs(trade2.executed_quantity) == pytest.approx(Decimal(4.443718014763590029))

    # position should be closed successfully
    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_positions) == 1



