"""Run the manual test to open and close 1delta position.

How to run:

.. code-block:: shell

    export JSON_RPC_POLYGON=
    export PK=
    export TRADING_STRATEGY_API_KEY=
    python scripts/manual-test-1delta-short.py
"""

import datetime
import os
from pathlib import Path
from decimal import Decimal

import pandas as pd

from web3 import HTTPProvider, Web3
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, AssetType, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.universe_model import default_universe_options
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, load_trading_and_lending_data, translate_trading_pair
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaSimpleRoutingModel, OneDeltaRoutingState
from tradeexecutor.cli.bootstrap import create_execution_and_sync_model, create_web3_config, create_state_store
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder

logger = setup_logging()


client = Client.create_live_client(api_key=os.environ["TRADING_STRATEGY_API_KEY"])

# prepare strategy universe
chain = ChainId.polygon

pairs = [
    (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
]

reverses = [
    (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
    (ChainId.polygon, LendingProtocolType.aave_v3, "USDC"),
]

dataset = load_partial_data(
    client,
    execution_context=python_script_execution_context,
    time_bucket=TimeBucket.d1,
    pairs=pairs,
    universe_options=default_universe_options,
    start_at=pd.Timestamp("2023-10-01"),
    end_at=pd.Timestamp("2023-11-23"),
    lending_reserves=reverses,
)

# Convert loaded data to a trading pair universe
strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

# setup execution and sync model
web3config = create_web3_config(
    json_rpc_binance=None,
    json_rpc_polygon=os.environ["JSON_RPC_POLYGON"],
    json_rpc_avalanche=None,
    json_rpc_ethereum=None,
    json_rpc_anvil=None,
    json_rpc_arbitrum=None,
)
web3config.choose_single_chain()
execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
    asset_management_mode=AssetManagementMode.hot_wallet,
    private_key=os.environ["PK"],
    web3config=web3config,
    confirmation_timeout=datetime.timedelta(seconds=60),
    confirmation_block_count=2,
    max_slippage=0.0025,
    min_gas_balance=0.1,
    vault_address=None,
    vault_adapter_address=None,
    vault_payment_forwarder_address=None,
    routing_hint=TradeRouting.one_delta_polygon_usdc,
)

store = create_state_store(Path("state/manual_test_1delta.json"))

if store.is_pristine():
    state = store.create("manual_test_1delta")
else:
    state = store.load()

usdc = fetch_erc20_details(sync_model.web3, "0x2791bca1f2de4661ed88a30c99a7a9449aa84174")
routing_model = OneDeltaSimpleRoutingModel(
    address_map={
        "one_delta_broker_proxy": "0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
        "aave_v3_pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        "aave_v3_data_provider": "0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654",
        "aave_v3_oracle": "0xb023e699F5a33916Ea823A16485e259257cA8Bd1",
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
        "position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
    },
    allowed_intermediary_pairs={},
    reserve_token_address=usdc.address.lower(),
)

# sync reserve
asset_usdc = AssetIdentifier(
    ChainId.polygon.value,
    usdc.contract.address,
    usdc.symbol,
    usdc.decimals,
)
sync_model.setup_all(state, [asset_usdc])

# init position manager
position_manager = PositionManager(
    datetime.datetime.utcnow(),
    strategy_universe,
    state,
    pricing_model_factory(
        execution_model,
        strategy_universe,
        routing_model,
    ),
)

# open short
pair_universe = strategy_universe.universe.pairs
pair = pair_universe.get_single()
trades = position_manager.open_short(pair, Decimal(0.5), leverage=1.1)

logger.info("Opening short trade: %s", trades[0])

# execute trades
tx_builder = sync_model.create_transaction_builder()
routing_state = OneDeltaRoutingState(pair_universe, tx_builder)

state.start_execution_all(datetime.datetime.utcnow(), trades)
routing_model.execute_trades_internal(pair_universe, routing_state, trades)
execution_model.broadcast_and_resolve_old(state, trades, stop_on_execution_failure=True)

assert trades[0].is_success()
