"""Check short flags esp. when closing the last short of the protocol."""
import datetime
import os
import shutil
from decimal import Decimal

import pandas as pd
import pytest as pytest
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket

from eth_defi.token import TokenDetails
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.state.trade import TradeFlag
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.runner import post_process_trade_decision
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import default_universe_options

pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)


@pytest.fixture()
def strategy_universe(
        chain_id,
        exchange_universe,
        asset_usdc,
        persistent_test_client
) -> TradingStrategyUniverse:

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "WMATIC"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-10-01"),
        end_at=pd.Timestamp("2023-10-30"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_from_dataset(dataset, asset_usdc.address)


@pytest.fixture
def wmatic_usdc_spot_pair(uniswap_v3_deployment, asset_usdc, asset_wmatic) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_wmatic,
        asset_usdc,
        "0xa374094527e1673a86de625aa59517c5de346d32",
        uniswap_v3_deployment.factory.address,
        fee=0.0005,
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
    weth_usdc_spot_pair: TradingPairIdentifier,
    wmatic_usdc_spot_pair: TradingPairIdentifier,
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

    post_process_trade_decision(state, unit_test_execution_context, trades)

    # Only last trade has close_protocol_last
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


