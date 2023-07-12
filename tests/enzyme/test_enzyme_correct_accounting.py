"""Test Enzyme accounting corrections can be applied."""
import datetime
import secrets
from decimal import Decimal

import pytest
import flaky
from eth_account import Account
from hexbytes import HexBytes

from eth_defi.enzyme.integration_manager import IntegrationManagerActionId
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.asset import get_relevant_assets
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
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
        weth_usdc_trading_pair) -> UniswapV2SimpleRoutingModel:

    # Allowed exchanges as factory -> router pairs
    factory_router_map = {
        uniswap_v2.factory.address: (uniswap_v2.router.address, uniswap_v2.init_code_hash),
    }

    # Three way ETH quoted trades are routed thru WETH/USDC pool
    allowed_intermediary_pairs = {
        weth_asset.address: weth_usdc_trading_pair.pool_address
    }

    return UniswapV2SimpleRoutingModel(
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


@flaky.flaky()
def test_enzyme_no_accounting_errors(
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
):
    """Check we get no accounting errors if there are one

    - Open one trading position

    - Accounting errors check should give us a clean psas
    """

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash = vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

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
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.get_total_equity() == pytest.approx(500)

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    # Check we have balance
    assert usdc.functions.balanceOf(tx_builder.get_erc_20_balance_address()).call() == 500 * 10**6

    # Now make a trade
    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
    )

    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(500),
        execute=False,
        slippage_tolerance=0.999,
    )

    trader.execute_trades_simple([trade], broadcast=True)

    assert weth.functions.balanceOf(vault.vault.address).call() > 0

    assets = get_relevant_assets(pair_universe, state.portfolio.get_reserve_assets(), state)
    balances = list(sync_model.fetch_onchain_balances(assets))

    # Should be all-in to ETH with some dust error
    assert len(balances) == 2
    b = balances[0]
    assert b.asset == usdc_asset
    assert b.amount == pytest.approx(Decimal("0.000001"))

    b = balances[1]
    assert b.asset == weth_asset
    assert b.amount == pytest.approx(Decimal("0.310787860635789571"))


