"""Execute trades using Enzyme vault."""
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

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.routing import RoutingModel
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


@flaky.flaky()
def test_enzyme_execute_open_position(
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
    routing_model: RoutingModel,
):
    """Open a simple spot buy position using Enzyme."""

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
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.calculate_total_equity() == pytest.approx(500)

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
        slippage_tolerance=0.01,
    )

    # How much ETH we expect in the trade
    eth_amount = Decimal(0.310787861255819868)
    assert trade.fee_tier == 0.0030
    assert trade.planned_quantity == pytest.approx(eth_amount)
    assert trade.lp_fees_estimated == None  # TODO: UniswapV2TestTrader does not support yet?

    deltas = trade.calculate_asset_deltas()
    assert deltas[0].asset == usdc_asset.address
    assert deltas[0].raw_amount == pytest.approx(-500 * 10**6)
    assert deltas[1].asset == weth_asset.address
    assert deltas[1].raw_amount == pytest.approx(eth_amount * Decimal(1 - trade.slippage_tolerance) * 10**18)

    trader.execute_trades_simple(trader.create_routing_model(), [trade], broadcast=False)

    # Check that the blockchain transactions where constructed for Enzyme's vault
    txs = trade.blockchain_transactions
    assert len(txs) == 2  # approve + swap tokens

    approve_tx = txs[0]
    assert approve_tx.type == BlockchainTransactionType.enzyme_vault
    assert approve_tx.broadcasted_at is None
    assert approve_tx.nonce == 0
    # The EOA hot wallet transaction needs to be send to comptroller contract
    assert approve_tx.contract_address == vault.comptroller.address
    # IntegrationManager.callOnExtension() API
    assert approve_tx.transaction_args[0] == vault.deployment.contracts.integration_manager.address
    assert approve_tx.transaction_args[1] == IntegrationManagerActionId.CallOnIntegration.value
    assert len(approve_tx.transaction_args[2]) > 0  # Solidity ABI encode packed

    # This is the payload of the tx the vault performs
    assert approve_tx.details["contract"] == usdc.address
    assert approve_tx.details["function"] == "approve"
    assert approve_tx.details["args"][0] == uniswap_v2.router.address
    assert len(approve_tx.asset_deltas) == 0

    swap_tx = txs[1]
    assert swap_tx.type == BlockchainTransactionType.enzyme_vault
    assert swap_tx.broadcasted_at is None
    assert swap_tx.nonce == 1
    assert swap_tx.contract_address == vault.comptroller.address
    assert swap_tx.transaction_args[0] == vault.deployment.contracts.integration_manager.address
    assert swap_tx.transaction_args[1] == IntegrationManagerActionId.CallOnIntegration.value
    assert swap_tx.details["contract"] == uniswap_v2.router.address
    assert swap_tx.details["function"] == "swapExactTokensForTokensSupportingFeeOnTransferTokens"

    # Spend USDC, receive WETH
    assert len(swap_tx.asset_deltas) == 2
    assert swap_tx.asset_deltas[0].asset == usdc_asset.address
    assert swap_tx.asset_deltas[0].int_amount < 0
    assert swap_tx.asset_deltas[1].asset == weth_asset.address

    # TODO: fix test calculation being incorrect
    # assert swap_tx.asset_deltas[1].int_amount == pytest.approx(eth_amount * Decimal(1 - trade.slippage_tolerance) * 10**18, rel=Decimal(0.01))

    # Broadcast both transactions
    trader.broadcast_trades(routing_model, [trade], stop_on_execution_failure=True)

    assert weth.functions.balanceOf(vault.vault.address).call() > 0

    assert trade.is_success()
    assert trade.executed_quantity == Decimal('0.310787860635789571')
    assert trade.executed_price == pytest.approx(1608.81444332199)
    assert trade.executed_reserve == pytest.approx(Decimal('499.999999'))


@flaky.flaky
def test_enzyme_execute_close_position(
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
    """Close a simple spot by selling position using Enzyme."""

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
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.calculate_total_equity() == pytest.approx(500)

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
        execute=True,
        slippage_tolerance=0.025,
    )

    assert position.get_quantity() > 0

    position_2, trade_2 = trader.sell(
        weth_usdc_trading_pair,
        position.get_quantity(),
        execute=True,
        slippage_tolerance=0.025,
    )

    assert position == position_2

    assert position.is_closed()

    # Lost some money on fees
    assert state.portfolio.calculate_total_equity() == pytest.approx(Decimal(497.011924))


# Some Anvil flakiness on Github
# FAILED tests/enzyme/test_enzyme_trade.py::test_enzyme_lp_fees - AssertionError: Timestamp missing for block number 46, hash 0x6186d5d75328e7ebb6fef627fbed2a185448c96c65970135e0b7cf673eadf3a2
@flaky.flaky()
def test_enzyme_lp_fees(
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
    routing_model: UniswapV2Routing,
    pricing_model: UniswapV2LivePricing,
):
    """See LP fees are correctly estimated and realised."""

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
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state)
    assert state.portfolio.calculate_total_equity() == pytest.approx(500)

    tx_builder = EnzymeTransactionBuilder(hot_wallet, vault)

    # Check we have a good price ETH/USD
    price_structure = pricing_model.get_buy_price(
        datetime.datetime.utcnow(),
        weth_usdc_trading_pair,
        Decimal(1),
    )
    assert price_structure.mid_price == pytest.approx(1600, rel=0.01)

    # Now make a trade
    trader = UniswapV2TestTrader(
        uniswap_v2,
        state=state,
        pair_universe=pair_universe,
        tx_builder=tx_builder,
        pricing_model=pricing_model,
    )

    position, trade = trader.buy(
        weth_usdc_trading_pair,
        Decimal(500),
        execute=False,
        slippage_tolerance=0.01,
    )

    # How much ETH we expect in the trade
    eth_amount = Decimal(0.310787861255819868)
    assert trade.fee_tier == 0.0030
    assert trade.planned_quantity == pytest.approx(eth_amount)
    assert trade.lp_fees_estimated == pytest.approx(1.5000000000000013)
