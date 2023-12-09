"""Test live routing of combined Uniswap v2, v3 spot and 1delta leveraged positions."""
import datetime
import os
import shutil
from decimal import Decimal

import pytest as pytest

from eth_defi.token import TokenDetails
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.state.trade import TradeFlag
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


def test_short_flags(
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricing,
    asset_usdc: AssetIdentifier,
    asset_weth: AssetIdentifier,
    asset_wmatic: AssetIdentifier,
    wmatic_usdc_spot_pair: TradingPairIdentifier,
    weth_usdc_spot_pair: TradingPairIdentifier,
    weth_usdc_shorting_pair: TradingPairIdentifier,
    execution_model: EthereumExecution,
    weth: TokenDetails,
    vweth: TokenDetails,
    ausdc: TokenDetails,
):
    """Check that short flags.

    - Open and close two parallel positions on 1delta

    - See that the trades are correctly flagged
    """

    routing_model = generic_routing_model

    sync_model = HotWalletSyncModel(
        web3,
        hot_wallet,
    )

    state = State()
    sync_model.sync_initial(state)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves=[asset_usdc])

    assert state.portfolio.get_reserve_position(asset_usdc).quantity == Decimal('10_000')

    # Setup routing state for the approvals of this cycle
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        strategy_universe,
        state,
        generic_pricing_model
    )

    # Trade on Quickswap spot
    trades = position_manager.open_short(
        wmatic_usdc_spot_pair,
        100.0,
    )

    # Trade on Quickswap spot
    trades += position_manager.open_short(
        weth_usdc_spot_pair,
        100.0,
    )

    assert TradeFlag.open in trades[0].flags
    assert TradeFlag.open in trades[1].flags
    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )
    assert all([t.is_success() for t in trades])

    # Close all positions
    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        strategy_universe,
        state,
        generic_pricing_model
    )
    trades = position_manager.close_all()

    assert TradeFlag.close in trades[0].flags
    assert TradeFlag.close in trades[1].flags
    assert TradeFlag.close_protocol_last not in trades[0].flags
    assert TradeFlag.close_protocol_last in trades[1].flags

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )


