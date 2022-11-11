from typing import List, Dict, Optional
from dataclasses import dataclass

import datetime
import pandas as pd
from pandas_ta.overlap import ema

import logging

from tradeexecutor.visual.benchmark import visualise_benchmark
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.strategy_module import StrategyType, TradeRouting, ReserveCurrency
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradeexecutor.strategy.trading_strategy_universe import load_all_data, TradingStrategyUniverse, \
    load_pair_data_for_single_exchange
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradingstrategy.client import Client
import datetime
from tradeexecutor.visual.single_pair import visualise_single_pair

# give it trade_routing, trading_strategy_cycle, candle_time_bucket, chain_id, exchange, pair, slow_ema_count, fast_ema_count, start_at, end_at, initial_deposit, position_size, batch_size 

@dataclass
class Generalized_strategy:
    decide_trades: callable
    trading_strategy_engine_version: str
    trading_strategy_type: StrategyType
    trade_routing: TradeRouting
    trading_strategy_cycle: CycleDuration
    reserve_currency: ReserveCurrency
    candle_time_bucket: TimeBucket
    chain_id: ChainId
    exchange_slug: str
    trading_pair_ticker: tuple[str, str]
    position_size: float
    batch_size: int
    slow_ema_candle_count: int
    fast_ema_candle_count: int
    start_at: datetime
    end_at: datetime
    initial_deposit: int

    def execute(self) -> None:
        """Execute entire backtesting procedure
            :param decide_trades:
            A function that decides whether to make a trade each cycle
        """

        def create_trading_universe(
            ts: datetime.datetime,
            client: Client,
            execution_context: ExecutionContext,
            candle_time_frame_override: Optional[TimeBucket]=None,
        ) -> TradingStrategyUniverse:
            # Load all datas we can get for our candle time bucket
            dataset = load_pair_data_for_single_exchange(
                client,
                execution_context,
                self.candle_time_bucket,
                self.chain_id,
                self.exchange_slug,
                [self.trading_pair_ticker],
                )


            # Filter down to the single pair we are interested in
            universe = TradingStrategyUniverse.create_single_pair_universe(
                dataset,
                self.chain_id,
                self.exchange_slug,
                self.trading_pair_ticker[0],
                self.trading_pair_ticker[1],
                )

            return universe

        client = Client.create_jupyter_client()

        self.state, self.universe, debug_dump = run_backtest_inline(
            name="BNB/USD EMA crossover example",
            start_at=self.start_at,
            end_at=self.end_at,
            client=client,
            cycle_duration=self.trading_strategy_cycle,
            decide_trades=self.decide_trades,
            create_trading_universe=create_trading_universe,
            initial_deposit=self.initial_deposit,
            reserve_currency=ReserveCurrency.busd,
            trade_routing=TradeRouting.pancakeswap_basic,
            log_level=logging.WARNING,
        )

        trade_count = len(list(self.state.portfolio.get_all_trades()))
        print(f"Backtesting completed, backtested strategy made {trade_count} trades")

        
    
    def show_pair(self):
        self.figure = visualise_single_pair(
            self.state,
            self.universe.universe.candles,
            start_at=self.start_at,
            end_at=self.end_at)
        self.figure.show()
    
    def show_benchmark(self):
        traded_pair = self.universe.universe.pairs.get_single()
        fig = visualise_benchmark(
            self.state.name,
            portfolio_statistics=self.state.stats.portfolio,
            all_cash=self.state.portfolio.get_initial_deposit(),
            buy_and_hold_asset_name=traded_pair.base_token_symbol,
            buy_and_hold_price_series=self.universe.universe.candles.get_single_pair_data()["close"],
            start_at=self.start_at,
            end_at=self.end_at
        )

        fig.show()