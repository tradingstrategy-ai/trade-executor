"""Execute trades using Enzyme vault."""
import datetime
import secrets
from decimal import Decimal

import pytest
from eth_account import Account
from hexbytes import HexBytes

from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
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


def test_enzyme_execute_open_position(
    web3: Web3,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    usdc_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
):
    """Open a simple spot buy position using Enzyme."""

    reorg_mon = create_reorganisation_monitor(web3)

    tx_hash =vault.vault.functions.addAssetManagers([hot_wallet.address]).transact({"from": user_1})
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

    # Now make a trade
    trader = UniswapV2TestTrader(
        web3,
        uniswap_v2,
        hot_wallet=tx_builder.hot_wallet,
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

    deltas = trade.calculate_asset_deltas()
