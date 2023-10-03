import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.lending import LendingCandleUniverse
from tradingstrategy.exchange import Exchange

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code



def create_synthetic_single_pair_universe(
    candles: pd.DataFrame,
    chain_id: ChainId, 
    exchange: Exchange,
    time_bucket: TimeBucket,
    pair: TradingPairIdentifier,
    lending_candles: LendingCandleUniverse = None,
) -> TradingStrategyUniverse:
    """Creates a synthetic universe with a single pair and a single chain.
    
    :param candles:
        A pandas DataFrame containing the candles for the pair.
        
        Must have the following columns:
            - timestamp
            - open
            - high
            - low
            - close
            - pair_id
    
        Can be generated using `generate_ohlcv_candles` from `tradeexecutor/testing/synthetic_price_data.py`.
    
    :param chain_id:
        Chain ID to use for the universe.

    :param exchange:
        Exchange to use for the universe. Can be generated using `generate_exchange` from `tradeexecutor/testing/synthetic_exchange_data.py`.
    
    :param time_bucket:
        Time bucket to use for the universe.

    :param pair:
        Pair to use for the universe.
    """


    assert type(candles) == pd.DataFrame, "candles must be a pandas DataFrame"
    assert type(chain_id) == ChainId, "chain_id must be of type ChainId"
    assert type(time_bucket) == TimeBucket, "time_bucket must be of type TimeBucket"
    assert type(pair) == TradingPairIdentifier, "pair must be of type TradingPairIdentifier"

    # Set up fake assets
    pair_universe = create_pair_universe_from_code(chain_id, [pair])

    # Load candles for backtesting
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
        lending_candles=lending_candles,
    )

    return TradingStrategyUniverse(
        universe=universe,
        reserve_assets=[pair.quote],
        backtest_stop_loss_time_bucket=time_bucket,
        backtest_stop_loss_candles=candle_universe,
    )