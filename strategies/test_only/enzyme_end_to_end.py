"""Dummy strategy used in Enzyme end-to-end tests.

"""
import datetime
from typing import Dict, List

import pandas as pd

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.pair import DEXPair, HumanReadableTradingPairDescription
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reserve_currency import ReserveCurrency

trading_strategy_engine_version = "0.1"
trading_strategy_type = StrategyType.managed_positions
trade_routing = TradeRouting.user_supplied_routing_model
trading_strategy_cycle = CycleDuration.cycle_1s
reserve_currency = ReserveCurrency.usdc


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model, default_slippage_tolerance=0.02)

    pair = universe.pairs.get_single()

    assert pair.pair_id > 0

    cash = state.portfolio.get_current_cash()

    cycle_number = cycle_debug_data["cycle"]

    trades = []

    # For odd seconds buy, for even seconds sell
    if cycle_number % 2 == 0:
        # buy on even days
        if not position_manager.is_any_open():
            position_size = 0.10
            buy_amount = cash * position_size
            trades += position_manager.open_1x_long(pair, buy_amount)
    else:
        # sell on odd days
        if position_manager.is_any_open():
            trades += position_manager.close_all()

    return trades


def create_trading_universe(
        ts: datetime.datetime,
        client: BaseClient,
        execution_context: ExecutionContext,
        universe_options: UniverseOptions,
):
    assert isinstance(client, UniswapV2MockClient), f"Looks like we are not running on EVM testing backend. Got: {client}"

    # Load exchange and pair data for a single pair
    dataset = load_all_data(
        client,
        TimeBucket.not_applicable,
        execution_context,
        universe_options,
    )

    # Create a trading universe for our test EVM backend Uniswap v2 deployment
    # assert len(dataset.pairs) == 1
    # use chain id and exchange slug from the first pair
    pair_data = dataset.pairs.iloc[0]
    pair: DEXPair = DEXPair.from_dict(pair_data.to_dict())

    # gets list of pair tickers and also reserve asset pair ticker
    reserve_asset_address = client.get_default_quote_token_address()
    reserve_asset_pair_ticker = None
    pairs: HumanReadableTradingPairDescription = []
    for row in dataset.pairs.itertuples():
        assert row.chain_id == pair.chain_id, "All pairs must be on the same chain"
        assert row.exchange_slug == pair.exchange_slug, "All pairs must be on the same exchange"
        
        pairs.append([row.base_token_symbol, row.quote_token_symbol])

        # find reserve asset ticker
        if reserve_asset_address in {row.token0_address, row.token1_address}:
            reserve_asset_pair_ticker = (row.base_token_symbol, row.quote_token_symbol)

    assert reserve_asset_pair_ticker is not None, "Could not find reserve asset ticker"

    universe = TradingStrategyUniverse.create_limited_pair_universe(
        dataset,
        pair.chain_id,
        pair.exchange_slug,
        pairs,
        reserve_asset_pair_ticker,
    )
    return universe