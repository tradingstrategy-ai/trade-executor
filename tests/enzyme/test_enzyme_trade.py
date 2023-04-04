"""Execute trades using Enzyme vault."""
import datetime
from _decimal import Decimal

import pytest
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.testing.ethereumtrader_uniswap_v2 import UniswapV2TestTrader


def test_enzyme_execute_open_position(
    web3: Web3,
    deployer: HexAddress,
    enzyme_vault_contract: Contract,
    vault_comptroller_contract: Contract,
    usdc: Contract,
    usdc_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_pair: TradingPairIdentifier,
):
    """Open a simple spot buy position using Enzyme."""

    reorg_mon = create_reorganisation_monitor(web3)

    sync_model = EnzymeVaultSyncModel(
        web3,
        enzyme_vault_contract.address,
        reorg_mon,
    )

    state = State()
    sync_model.sync_initial(state)

    # Make two deposits from separate parties
    usdc.functions.transfer(user_1, 500 * 10**6).transact({"from": deployer})
    usdc.functions.approve(vault_comptroller_contract.address, 500 * 10**6).transact({"from": user_1})
    vault_comptroller_contract.functions.buyShares(500 * 10**6, 1).transact({"from": user_1})

    # Strategy has its reserve balances updated
    # assert state.portfolio.get_total_equity() == pytest.approx(1200)

    # Now make a trade
    # trader = UniswapV2TestTrader(web3, uniswap_v2, hot_wallet, state, pair_universe)
    # position, trade = trader.buy(weth_usdc_pair, Decimal(500))

