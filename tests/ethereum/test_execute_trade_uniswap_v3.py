"""Test trading and portfolio management against a Uniswap pool in Ethereum Tester environment."""

import datetime
import secrets
from decimal import Decimal
from typing import List

import pytest
from eth_account import Account
from eth_typing import HexAddress
from hexbytes import HexBytes

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradingstrategy.pair import PandasPairUniverse
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.token import create_token
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, deploy_uniswap_v3, deploy_pool , add_liquidity
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from eth_defi.uniswap_v3.utils import get_default_tick_range

from tradeexecutor.ethereum.execution import get_held_assets
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import get_current_price
from tradeexecutor.ethereum.universe import create_pair_universe
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.ethereum.balance_update import apply_reserve_update_events
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.testing.ethereumtrader_uniswap_v3 import UniswapV3TestTrader
from tradeexecutor.testing.unit_test_trader import UnitTestTrader



WETH_USDC_FEE = 0.003
AAVE_USDC_FEE = 0.003

WETH_USDC_FEE_RAW = 3000
AAVE_USDC_FEE_RAW = 3000


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


@pytest.fixture
def aave_token(web3, deployer: HexAddress) -> Contract:
    """Create AAVE with 10M supply."""
    token = create_token(web3, deployer, "Fake Aave coin", "AAVE", 10_000_000 * 10**18, 18)
    return token


@pytest.fixture()
def uniswap_v3(web3, deployer) -> UniswapV3Deployment:
    """Uniswap v2 deployment."""
    deployment = deploy_uniswap_v3(web3, deployer)
    return deployment

@pytest.fixture()
def price_helper(uniswap_v3):
    return UniswapV3PriceHelper(uniswap_v3)

@pytest.fixture
def weth_token(uniswap_v3: UniswapV3Deployment) -> Contract:
    """Mock some assets"""
    return uniswap_v3.weth


@pytest.fixture
def asset_usdc(usdc_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, usdc_token.address, usdc_token.functions.symbol().call(), usdc_token.functions.decimals().call())


@pytest.fixture
def asset_weth(weth_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, weth_token.address, weth_token.functions.symbol().call(), weth_token.functions.decimals().call())


@pytest.fixture
def asset_aave(aave_token, chain_id) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(chain_id, aave_token.address, aave_token.functions.symbol().call(), aave_token.functions.decimals().call())


@pytest.fixture
def aave_usdc_uniswap_trading_pair(web3, deployer, uniswap_v3, aave_token, usdc_token) -> HexAddress:
    """AAVE-USDC pool with 200k liquidity. Fee of 0.1%"""
    min_tick, max_tick = get_default_tick_range(AAVE_USDC_FEE_RAW)

    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=aave_token,
        token1=usdc_token,
        fee=AAVE_USDC_FEE_RAW
    )

    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 AAVE liquidity
        amount1=200_000 * 10**6,  # 200k USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick
    )
    return pool_contract.address


@pytest.fixture
def weth_usdc_uniswap_trading_pair(web3, deployer, uniswap_v3, weth_token, usdc_token) -> HexAddress:
    """ETH-USDC pool with 1.7M liquidity."""
    min_tick, max_tick = get_default_tick_range(WETH_USDC_FEE_RAW)

    pool_contract = deploy_pool(
        web3,
        deployer,
        deployment=uniswap_v3,
        token0=weth_token,
        token1=usdc_token,
        fee=WETH_USDC_FEE_RAW
    )

    add_liquidity(
        web3,
        deployer,
        deployment=uniswap_v3,
        pool=pool_contract,
        amount0=1000 * 10**18,  # 1000 ETH liquidity
        amount1=1_700_000 * 10**6,  # 1.7M USDC liquidity
        lower_tick=min_tick,
        upper_tick=max_tick
    )
    return pool_contract.address


@pytest.fixture
def weth_usdc_pair(uniswap_v3, weth_usdc_uniswap_trading_pair, asset_usdc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_weth,
        asset_usdc,
        weth_usdc_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee = WETH_USDC_FEE
    )


@pytest.fixture
def aave_usdc_pair(uniswap_v3, aave_usdc_uniswap_trading_pair, asset_usdc, asset_aave) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_aave,
        asset_usdc,
        aave_usdc_uniswap_trading_pair,
        uniswap_v3.factory.address,
        fee = AAVE_USDC_FEE
    )


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """Timestamp of action started"""
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


@pytest.fixture
def state(web3, hot_wallet, asset_usdc) -> State:
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


@pytest.fixture()
def pair_universe(web3, weth_usdc_pair, aave_usdc_pair) -> PandasPairUniverse:
    return create_pair_universe(web3, None, [weth_usdc_pair, aave_usdc_pair])


@pytest.fixture()
def tx_builder(web3, hot_wallet) -> HotWalletTransactionBuilder:
    tx_builder = HotWalletTransactionBuilder(
        web3,
        hot_wallet,
    )
    return tx_builder


@pytest.fixture()
def ethereum_trader(
        web3: Web3,
        uniswap_v3: UniswapV3Deployment,
        hot_wallet: HotWallet,
        state: State,
        pair_universe: PandasPairUniverse,
        tx_builder: HotWalletTransactionBuilder,
) -> UniswapV3TestTrader:
    return UniswapV3TestTrader(uniswap_v3, state, pair_universe, tx_builder)


def test_execute_trade_instructions_buy_weth(
    web3: Web3,
    state: State,
    pair_universe: PandasPairUniverse,
    uniswap_v3: UniswapV3Deployment,
    hot_wallet: HotWallet,
    usdc_token: AssetIdentifier,
    weth_token: AssetIdentifier,
    weth_usdc_pair: TradingPairIdentifier,
    start_ts: datetime.datetime,
    price_helper: UniswapV3PriceHelper,
    ethereum_trader: UniswapV3TestTrader
):
    """Sync reserves from one deposit."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    # Buy 500 USDC worth of WETH
    trader = UnitTestTrader(state)

    buy_amount = 500

    # swap from quote to base (usdc to weth)
    path = [usdc_token.address, weth_token.address]
    fees = [WETH_USDC_FEE_RAW]

    # Estimate price
    raw_assumed_quantity = price_helper.get_amount_out(buy_amount * 10 ** 6, path, fees)

    assumed_quantity = Decimal(raw_assumed_quantity) / Decimal(10**18)
    assert assumed_quantity == pytest.approx(Decimal(0.293149332386944192))

    position, trade = trader.prepare_buy(weth_usdc_pair, assumed_quantity, 1700)
    assert state.portfolio.calculate_total_equity() == pytest.approx(10000.0)
    assert trade.get_status() == TradeStatus.planned

    ethereum_trader.execute_trades_simple([trade])

    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(1705.613701292989)
    assert trade.executed_quantity == pytest.approx(Decimal(0.292184487391376249))
    assert trade.lp_fees_paid == pytest.approx(1.495061595)
    assert trade.native_token_price == 0.0
    # assert trade.cost_of_gas == pytest.approx(Decimal('0.000154891504185768'))


def test_execute_trade_instructions_buy_weth_with_tester(
        web3: Web3,
        state: State,
        uniswap_v3: UniswapV3Deployment,
        hot_wallet: HotWallet,
        pair_universe,
        weth_usdc_pair: TradingPairIdentifier,
        start_ts: datetime.datetime,
        tx_builder
):
    """Same as above but with the tester class.."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    # Buy 500 USDC worth of WETH
    trader = UniswapV3TestTrader(uniswap_v3, state, pair_universe, tx_builder)
    position, trade = trader.buy(weth_usdc_pair, Decimal(500))

    assert position.is_open()

    assert trade.planned_price == pytest.approx(1705.615346038114)
    assert trade.planned_quantity == pytest.approx(Decimal('0.293149332386944223'))

    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(1705.6153460473756)
    assert trade.executed_quantity == pytest.approx(Decimal('0.29314933179905376'))
    assert trade.lp_fees_paid == pytest.approx(1.499999997)
    assert trade.native_token_price == 0.0
    # assert trade.cost_of_gas == pytest.approx(Decimal('0.000154891504185768'))

    # Cash balance has been deducted
    assert portfolio.get_cash() == pytest.approx(9500.0)

    # Portfolio is correctly valued
    assert portfolio.calculate_total_equity() == pytest.approx(9999.999998997286)


def test_buy_sell_buy_with_tester(
    web3: Web3,
    state: State,
    uniswap_v3: UniswapV3Deployment,
    hot_wallet: HotWallet,
    pair_universe,
    weth_usdc_pair: TradingPairIdentifier,
    start_ts: datetime.datetime,
    tx_builder,
    weth_token: Contract,
):
    """Execute three trades on a position."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    #
    # 1. Buy 500 USDC worth of WETH
    #

    trader = UniswapV3TestTrader(uniswap_v3, state, pair_universe, tx_builder)
    position, trade = trader.buy(weth_usdc_pair, Decimal(500))

    assert position.is_open()
    assert trade.planned_price == pytest.approx(1705.615346038114)
    assert trade.planned_quantity == pytest.approx(Decimal('0.293149332386944223'))

    # Check onchain quantity is correct
    onchain_balance = weth_token.functions.balanceOf(hot_wallet.address).call()
    assert (onchain_balance / 10**18) == pytest.approx(0.293149332386944223)

    # Calculate price by hand
    real_price = 500 / 0.293149332386944223
    assert real_price == pytest.approx(1705.615346038114)

    assert trade.get_status() == TradeStatus.success
    assert trade.executed_quantity == pytest.approx(Decimal('0.29314933179905376'))
    assert trade.executed_price == pytest.approx(1705.615346038114)

    assert portfolio.get_cash() == pytest.approx(9500.0)
    assert portfolio.calculate_total_equity() == pytest.approx(9999.999998997286)

    #
    # 2. Sell all bought ETH
    #

    assert position.get_quantity() == pytest.approx(Decimal('0.29314933179905376'))
    position2, trade2 = trader.sell(weth_usdc_pair, position.get_quantity())

    # We get the same position object as in the first buy
    assert position2.position_id == position.position_id
    assert position2.get_quantity_old() == 0
    assert position2.is_closed()

    assert trade2.get_status() == TradeStatus.success
    assert trade2.executed_price == pytest.approx(1695.398495196584)
    assert trade2.executed_quantity == pytest.approx(Decimal('-0.29314933179905376'))
    assert trade2.lp_fees_paid == pytest.approx(1.491014808)

    assert portfolio.get_cash() == pytest.approx(9997.004936)
    assert portfolio.calculate_total_equity() == pytest.approx(9997.004936)

    #
    # 3. Buy ETH again as a regret buy
    # This will open a new position

    position3, trade3 = trader.buy(weth_usdc_pair, Decimal(500))

    assert position3.is_open()
    assert position3.position_id != position.position_id
    assert position3.get_quantity_old() == pytest.approx(Decimal('0.293148816843562091'))

    assert trade3.planned_price == pytest.approx(1705.618345602341)
    assert trade3.planned_quantity == pytest.approx(Decimal('0.293148816843562091'))
    assert trade3.executed_price == pytest.approx(1705.618345602341)
    assert trade3.executed_quantity == pytest.approx(Decimal('0.293148816843562091'))
    # assert trade3.cost_of_gas == pytest.approx(Decimal('0.000134461088522565'))
    assert trade3.lp_fees_paid == pytest.approx(1.5)

    # Double check See we can serialise state after all this
    dump = state.to_json()
    state2: State = State.from_json(dump)
    assert len(state2.portfolio.closed_positions) == 1


def test_buy_buy_sell_sell_tester(
        web3: Web3,
        state: State,
        uniswap_v3: UniswapV3Deployment,
        hot_wallet: HotWallet,
        weth_usdc_pair: TradingPairIdentifier,
        pair_universe,
        tx_builder,
        start_ts: datetime.datetime):
    """Execute four trades on the same position."""

    portfolio = state.portfolio

    # We have everything in cash
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000

    #
    # 1. Buy 500 USDC worth of WETH
    # 2. Buy 500 USDC worth of WETH
    #

    trader = UniswapV3TestTrader(uniswap_v3, state, pair_universe, tx_builder)
    position1, trade1 = trader.buy(weth_usdc_pair, Decimal(500))
    position2, trade2 = trader.buy(weth_usdc_pair, Decimal(500))

    # 1000 USDC for 1700 USD/ETH
    weth_holding = position2.get_quantity_old()
    assert weth_holding == pytest.approx(Decimal('0.586126840906346334'))

    # Now liquidate the portfolio
    sell_quantity_1 = weth_holding / 2
    sell_quantity_2 = weth_holding - sell_quantity_1

    position3, trade3 = trader.sell(weth_usdc_pair, sell_quantity_1)
    position4, trade4 = trader.sell(weth_usdc_pair, sell_quantity_2)

    assert trade4.lp_fees_paid == pytest.approx(1.49058033)
    # assert trade4.cost_of_gas == pytest.approx(Decimal('0.00010607491980141'))

    assert position4.is_closed()

    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1

    # We have everything in cash and lost some USDC in trading fees
    assert portfolio.calculate_total_equity() == pytest.approx(9994.010745)
    assert portfolio.get_cash() == pytest.approx(9994.010745)


def test_two_parallel_positions(
        web3: Web3,
        state: State,
        uniswap_v3: UniswapV3Deployment,
        hot_wallet: HotWallet,
        weth_usdc_pair: TradingPairIdentifier,
        aave_usdc_pair: TradingPairIdentifier,
        asset_aave,
        asset_weth,
        asset_usdc,
        pair_universe,
        tx_builder,
        start_ts: datetime.datetime):
    """Execute four trades on two positions at the same time."""

    portfolio = state.portfolio

    # We have everything in cash and initial assumptions on the price
    assert portfolio.calculate_total_equity() == 10_000
    assert portfolio.get_cash() == 10_000
    assert get_current_price(web3, uniswap_v3, weth_usdc_pair) == pytest.approx(1693.211867)
    assert get_current_price(web3, uniswap_v3, aave_usdc_pair) == pytest.approx(199.201396)
    assert hot_wallet.current_nonce == 0

    #
    # 1. Buy 500 USDC worth of WETH at 1700 USD
    # 2. Buy 500 USDC worth of AAVE at 200 USD
    #

    trader = UniswapV3TestTrader(uniswap_v3, state, pair_universe, tx_builder)
    position1, trade1 = trader.buy(weth_usdc_pair, Decimal(500), execute=False)
    position2, trade2 = trader.buy(aave_usdc_pair, Decimal(500), execute=False)

    assert position1.position_id == 1
    assert position2.position_id == 2

    assert len(portfolio.open_positions) == 2

    # Execute both trades
    trader.execute_trades_simple([trade1, trade2])
    assert hot_wallet.current_nonce == 3

    assert position1.get_quantity_old() == pytest.approx(Decimal('0.29314933179905376'))
    assert position2.get_quantity_old() == pytest.approx(Decimal('2.486302890046558723'))
    assert position1.get_value() == pytest.approx(499.999998997285)
    assert position2.get_value() == pytest.approx(500.00000000000006)
    assert portfolio.calculate_total_equity() == pytest.approx(9999.999998997286)
    assert portfolio.get_cash() == pytest.approx(9000.0)

    balances = get_held_assets(web3, hot_wallet.address, [asset_usdc, asset_aave, asset_weth])
    assert balances[asset_usdc.address] == Decimal("9000.000001")
    assert balances[asset_aave.address] == Decimal('2.486302890046558723')
    assert balances[asset_weth.address] == Decimal('0.29314933179905376')

    #
    # 3. Sell all WETH
    # 4. Sell all AAVE
    #

    position3, trade3 = trader.sell(weth_usdc_pair, position1.get_quantity_old(), execute=False)
    position4, trade4 = trader.sell(aave_usdc_pair, position2.get_quantity_old(), execute=False)

    trader.execute_trades_simple([trade3, trade4])

    assert trade3.blockchain_transactions[0].nonce == 3
    assert trade4.blockchain_transactions[0].nonce == 5

    # assert trade3.cost_of_gas == pytest.approx(Decimal('0.000124291689741213'))
    assert trade3.lp_fees_paid == pytest.approx(1.49101480)

    # assert trade4.cost_of_gas == pytest.approx(Decimal('0.000128945834655299'))
    assert trade4.lp_fees_paid == pytest.approx(1.491024620999999)

    assert position3.position_id == 1
    assert position4.position_id == 2
    assert position3.get_quantity() == 0
    assert position4.get_quantity() == 0
    assert position3.is_closed()
    assert position4.is_closed()
    assert portfolio.calculate_total_equity() == pytest.approx(9994.013143)
    assert portfolio.get_cash() == pytest.approx(9994.013143)

    balances = get_held_assets(web3, hot_wallet.address, [asset_usdc, asset_aave, asset_weth])
    assert balances[asset_usdc.address] == pytest.approx(Decimal('9994.013144'))
    assert balances[asset_aave.address] == 0
    assert balances[asset_weth.address] == 0
