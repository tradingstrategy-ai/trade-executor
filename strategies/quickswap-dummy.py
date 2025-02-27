"""This is a test strategy that does nothing.

- It is designed to show how to use `trade-executor start` command
  to launch a live strategy

- It connects to RPC node, but does not do any trades,
  but idles and loads the data every 1h

How to run

- We have prepared a random private key with zero balance

- Get a Polygon JSON-RPC access. You can use free endpoint
  ``https://polygon-rpc.com``.

- We do not need any balance on the hot wallet address,
  as we do not do any trades and set ``MIN_GAS_BALANCE``,
  this example is just a demostration how to get the strategy started up.

- Create an empty working folder where we are going to run

- Create a file ``quickswap-dummy.py`` and paste the contents of this file

- Create empty folders ``state``, ``cache``, ``logs`` in the current working directory.
  ``trade-executor`` will write into these folders.

- Make sure you have an API key for ``TRADING_STRATEGY_API_KEY`` environment variable
  or command line argument.

Run using ``docker`` command:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="..."
    export PRIVATE_KEY=0xa8cc6222cbd6b0b31c2f8216ff905fde71f0575e302d8410796c620f06254b5d
    export STRATEGY_FILE=quickswap-dummy.py
    export ASSET_MANAGEMENT_MODE=hot_wallet
    export JSON_RPC_POLYGON="https://polygon-rpc.com"
    export MIN_GAS_BALANCE=0

    # Explanation
    # --tty: colour output
    # --interactive: allow abort with CTRL+C
    # -v, -w: map current working directory folders to docker
    # -e: pass environment variables
    # start: trade-executor internal subcommand

    docker \
        run \
        --tty \
        --interactive \
        -v `pwd`:`pwd` \
        -w `pwd` \
        -e PRIVATE_KEY \
        -e STRATEGY_FILE \
        -e ASSET_MANAGEMENT_MODE \
        -e JSON_RPC_POLYGON \
        -e TRADING_STRATEGY_API_KEY \
        -e MIN_GAS_BALANCE \
        ghcr.io/tradingstrategy-ai/trade-executor:latest \
        start

For ``docker-compose.yaml`` instructions please refer to the documentation.

You can also run without Docker, using the checked out Python application directly
in its Poetry environment:

.. code-block:: shell

    trade-executor \
        start \
        --trading-strategy-api-key="..." \
        --private-key=0xa8cc6222cbd6b0b31c2f8216ff905fde71f0575e302d8410796c620f06254b5d \
        --strategy-file=quickswap-dummy.py \
        --asset-management-mode=hot_wallet \
        --json-rpc-polygon="https://polygon-rpc.com" \
        --min-gas-balance=0

Or:

.. code-block:: shell

    export TRADING_STRATEGY_API_KEY="..."
    export PRIVATE_KEY=0xa8cc6222cbd6b0b31c2f8216ff905fde71f0575e302d8410796c620f06254b5d
    export STRATEGY_FILE=quickswap-dummy.py
    export ASSET_MANAGEMENT_MODE=hot_wallet
    export JSON_RPC_POLYGON="https://polygon-rpc.com"
    export MIN_GAS_BALANCE=0

    trade-executor start

The trade-executor should start, load some market data and eventually print at the end:

.. code-block:: text

    2023-10-23 15:53:15 apscheduler.scheduler                              INFO     Added job "ExecutionLoop.run_live.<locals>.live_cycle" to job store "default"
    2023-10-23 15:53:15 apscheduler.scheduler                              INFO     Added job "ExecutionLoop.run_live.<locals>.live_positions" to job store "default"
    2023-10-23 15:53:15 apscheduler.scheduler                              INFO     Added job "ExecutionLoop.run_live.<locals>.live_trigger_checks" to job store "default"
    2023-10-23 15:53:15 apscheduler.scheduler                              INFO     Scheduler started

"""


import datetime
from typing import List, Dict

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency

# Tell what trade execution engine version this strategy needs to use
trading_strategy_engine_version = "0.3"

# What kind of strategy we are running.
# This tells we are going to use
trading_strategy_type = StrategyType.managed_positions

# How our trades are routed.
# PancakeSwap basic routing supports two way trades with BUSD
# and three way trades with BUSD-BNB hop.
trade_routing = TradeRouting.quickswap_usdc

# Attempt to trade every 1h
trading_strategy_cycle = CycleDuration.cycle_1h

# Strategy keeps its cash in USDC
reserve_currency = ReserveCurrency.usdc

# The duration of the backtesting period
backtest_start = datetime.datetime(2020, 11, 1)
backtest_end = datetime.datetime(2023, 1, 31)

# Start with 10,000 USD
initial_cash = 10_000


def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
) -> List[TradeExecution]:
    # Never do any trades
    return []


def create_trading_universe(
        ts: datetime.datetime,
        client: Client,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    # Load ETH-USDC on Uni v3
    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    dataset = load_partial_data(
        client,
        execution_context=execution_context,
        time_bucket=TimeBucket.h1,
        universe_options=universe_options,
        pairs=pairs,
        required_history_period=datetime.timedelta(days=7),  # Back fetch candles for 7 days
    )

    # Filter down the dataset to the pairs we specified
    universe = TradingStrategyUniverse.create_from_dataset(dataset)

    return universe

