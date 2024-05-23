"""Realised profit w/redemption test."""

import datetime
import secrets
from decimal import Decimal

import pytest
from eth_account import Account
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from eth_typing import HexAddress

from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_sell_price
from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from tradingstrategy.pair import PandasPairUniverse
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator
from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import revalue_state, ValuationModel
from tradeexecutor.testing.ethereumtrader_uniswap_v2 import UniswapV2TestTrader


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def routing_model(
        uniswap_v2,
        usdc_asset,
        weth_asset,
        weth_usdc_trading_pair) -> UniswapV2Routing:

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        uniswap_v2.factory.address: (uniswap_v2.router.address, uniswap_v2.init_code_hash),
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        weth_asset.address: weth_usdc_trading_pair.pool_address
    }

    return UniswapV2Routing(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=usdc_asset.address,
    )


@pytest.fixture()
def pricing_model(
        web3,
        uniswap_v2,
        pair_universe: PandasPairUniverse,
        routing_model) -> UniswapV2LivePricing:

    pricing_model = UniswapV2LivePricing(
        web3,
        pair_universe,
        routing_model,
    )
    return pricing_model


@pytest.fixture()
def valuation_model(pricing_model):
    return UniswapV2PoolRevaluator(pricing_model)


def test_enzyme_redeemed_position_profit(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    pricing_model: PricingModel,
    valuation_model: ValuationModel,
    single_pair_strategy_universe: TradingStrategyUniverse,
):
    """Profit is correctly calculated for a position with redemptions.

    - Redeem from one open WETH position

    - Move the price, so that we make losses on the position

    - Close the position

    - Chec that realised PnL is correctly calculated for the position with redemptions
    """

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Starting price for 1 ETH
    starting_price = estimate_sell_price(uniswap_v2, weth, usdc, 1 * 10**18) / 1e6
    assert starting_price == pytest.approx(1582.577362)

    sync_model = EnzymeVaultSyncModel(
        web3,
        vault.address,
        reorg_mon,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": user_1})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    # Create open WETH/USDC position worth of 100

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
    )

    # Open $100 position
    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(100),
        execute=True,
        slippage_tolerance=0.0125,
    )
    assert trade.is_success()

    # Redeem 50%
    # Shares originally = 500
    # Redeem 250 shares
    tx_hash = vault.comptroller.functions.redeemSharesInKind(user_1, 250 * 10**18, [], []).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    events = sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert len(events) == 2  # redemption detected for two assts

    assert state.portfolio.open_positions[1].get_unrealised_profit_usd() == 0

    # Move the price
    # The pool is 1000 ETH / 1.7M USDC.
    # Deployer dumps 10 ETH to cause a price impact.
    weth.functions.approve(uniswap_v2.router.address, 1000 * 10**18).transact({"from": deployer})
    prepared_swap_call = swap_with_slippage_protection(
        uniswap_v2_deployment=uniswap_v2,
        recipient_address=deployer,
        base_token=usdc,
        quote_token=weth,
        amount_in=10 * 10**18,
        max_slippage=10_000,
    )
    tx_hash = prepared_swap_call.transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Price after movement
    moved_price = estimate_sell_price(uniswap_v2, weth, usdc, 1 * 10**18) / 1e6
    assert moved_price == pytest.approx(1359.153875)

    # Revalue positions
    revalue_state(state, datetime.datetime.utcnow(), valuation_model)

    # Because price went down we have unrealised PnL
    assert state.portfolio.open_positions[1].get_unrealised_profit_usd() == pytest.approx(-7.372047000000003)

    # Close the position
    position_manager = PositionManager(datetime.datetime.utcnow(), single_pair_strategy_universe, state, pricing_model)
    trades = position_manager.close_all()
    assert len(trades) == 1

    trader.execute_trades_simple(trader.create_routing_model(), trades)

    # Position has redemptions
    assert len(position.balance_updates) == 1

    # Portfolio has 250 + losses moved away
    reserve = state.portfolio.get_default_reserve_position()
    assert position.get_value() == 0
    assert reserve.get_value() == pytest.approx(242.627953)
    assert state.portfolio.get_total_equity() == pytest.approx(242.627953)

    assert position.get_realised_profit_usd() == pytest.approx(-7.372047000000003)
    # (1359.153875 - 1582.577362) / 1582.577362 ~= -14.1
    assert position.get_realised_profit_percent() == pytest.approx(-0.14744093999999996)
    assert position.get_total_profit_usd() == position.get_realised_profit_usd()
    assert position.get_total_profit_usd() == pytest.approx(-7.372047000000003)