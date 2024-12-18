"""ETH-BTC-USDC strategy variant, testing size risk."""
import datetime
import os

import pandas_ta
import pandas as pd

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.visualisation import PlotKind, PlotLabel, PlotShape
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_passthrouh, weight_by_1_slash_n
from tradeexecutor.utils.binance import create_binance_universe
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.lending import LendingReserveDescription, LendingProtocolType
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.groupeduniverse import resample_price_series

trading_strategy_engine_version = "0.5"


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "capacity-limited"  # Used in cache paths

    cycle_duration = CycleDuration.cycle_97h  # A very arbitrary number

    #
    # Trading universe setup
    #

    chain_id = ChainId.ethereum
    candle_time_bucket = TimeBucket.h1

    #
    # Strategy parameters
    #
    momentum_lookback_bars = 20 * 24  # Momentum based returns of this period (bars)
    minimum_mometum_threshold = 0.01  # % of minimum momentum before the asset is included in the basket
    allocation = 0.98  # How much cash to allocate to the portfolio
    max_assets_in_portfolio = 2  # Amount of assets in the basket
    minimum_rebalance_trade_threshold_usd = 10_000  # Min trade size we are willing to do
    stop_loss = None  # Are we using stop loss on open positions

    use_credit = True  # Do we use Aave lending pools for the extra yield
    aave_min_deposit_threshold_usd = 5000  # Do not deposit to Aave if the cash in hand is less than this (already deposited)

    #
    # Capacity limiting
    #
    per_position_cap_of_pool = 0.025  # Cap entires to 2.5% of the underlying pool size

    # We set this here to test this parameter,
    # but current backtesting setup always assumes 0% price impact
    max_price_impact = 0.03

    #
    # Live trading only
    #
    chain_id = ChainId.ethereum
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(hours=momentum_lookback_bars + 1)

    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2022, 5, 1)
    backtest_end = datetime.datetime(2023, 2, 1)
    initial_cash = 100_000_000
    use_binance = False  # Binance does not have depth/TVL data so we cannot use it



def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    """Switch between backtest and live trading pairs.

    Because the live trading DEX venues do not have enough history (< 2 years)
    for meaningful backtesting, we test with Binance CEX data.
    """

    use_binance = Parameters.use_binance

    if use_binance:
        assert not execution_mode.is_live_trading(), "Binance market data can be only used for backtesting"

    if use_binance:
        trading_pairs = [
             (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
             (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]

    else:
        trading_pairs = [
            (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),
            (ChainId.polygon, "quickswap", "WETH", "USDC", 0.0030),
        ]

    return trading_pairs



def get_lending_reserves(mode: ExecutionMode) -> list[LendingReserveDescription]:
    """Get lending reserves the strategy needs."""

    use_binance = Parameters.use_binance

    if use_binance or (not Parameters.use_credit):
        # Credit interest is not available on Binance
        return []
    else:
        # We use Aave v2 in backtesting (longer history)
        # and Aave v3 in live execution (more liquid market)
        if mode.is_backtesting():
            lending_reserves = [
                (ChainId.polygon, LendingProtocolType.aave_v2, "USDC"),
            ]
        else:
            lending_reserves = [
                (ChainId.polygon, LendingProtocolType.aave_v3, "USDC"),
            ]

    return lending_reserves


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    trading_pairs = get_strategy_trading_pairs(execution_context.mode)
    lending_reserves = get_lending_reserves(execution_context.mode)

    use_binance = trading_pairs[0][0] == ChainId.centralised_exchange

    if use_binance:
        # Backtesting - load Binance data
        strategy_universe = create_binance_universe(
            [f"{p[2]}{p[3]}" for p in trading_pairs],
            candle_time_bucket=Parameters.candle_time_bucket,
            start_at=universe_options.start_at,
            end_at=universe_options.end_at,
            trading_fee_override=Parameters.backtest_trading_fee,
            include_lending=False,
            forward_fill=True,
        )
    else:

        if execution_context.live_trading or execution_context.mode == ExecutionMode.preflight_check:
            # Live trading
            start_at, end_at = None, None
            required_history_period=Parameters.required_history_period
        else:
            # Create backtest trading universe
            required_history_period = None
            start_at=universe_options.start_at
            end_at=universe_options.end_at

        dataset = load_partial_data(
            client,
            execution_context=execution_context,
            time_bucket=Parameters.candle_time_bucket,
            pairs=trading_pairs,
            universe_options=universe_options,
            start_at=start_at,
            end_at=end_at,
            lending_reserves=lending_reserves,
            required_history_period=required_history_period,
            # We need TVL data for capacity estimations
            liquidity=True,
            liquidity_time_bucket=TimeBucket.d1,
        )

        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            dataset,
            forward_fill=True,
            reserve_asset="0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # USDC.e on Polygon
        )

    return strategy_universe


def momentum(close, momentum_lookback_bars) -> pd.Series:
    """Calculate momentum series to be used as a signal.

    This indicator is later processed in decide_trades() to a weighted alpha signal.

    :param momentum_lookback_bars:
        Calculate returns based on this many bars looked back
    """
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html#pandas.DataFrame.shift
    start_close = close.shift(momentum_lookback_bars)
    momentum = (close - start_close) / start_close
    return momentum


def create_indicators(
        timestamp: datetime.datetime | None,
        parameters: StrategyParameters,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext
):
    indicators = IndicatorSet()
    indicators.add(
        "momentum",
        momentum,
        {"momentum_lookback_bars": parameters.momentum_lookback_bars},
        IndicatorSource.close_price,
    )
    return indicators


def decide_trades(
        input: StrategyInput,
) -> list[TradeExecution]:
    #
    # Decidion cycle setup.
    # Read all variables we are going to use for the decisions.
    #
    parameters = input.parameters
    position_manager = input.get_position_manager()
    state = input.state
    timestamp = input.timestamp
    indicators = input.indicators
    strategy_universe = input.strategy_universe
    alpha_model = AlphaModel(timestamp)

    # Check all pairs and construct a signal for them
    for pair in strategy_universe.iterate_pairs():

        momentum = indicators.get_indicator_value("momentum", pair=pair)

        if momentum is None:
            # Data not yet available for this trading pair
            continue

        if momentum >= parameters.minimum_mometum_threshold:
            #
            # Set the signal as the momentum value
            #
            alpha_model.set_signal(
                pair,
                momentum,
                stop_loss=parameters.stop_loss,
            )
            position_manager.log(f"Signal {pair.get_ticker()}: {momentum}")

            # Our trading starts as soon as we have valid
            # monmentum for the first valid trading apir
            state.mark_ready(timestamp)

        else:
            position_manager.log(f"Signal discarded {pair.get_ticker()}: {momentum}")

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_total_equity() * parameters.allocation

    #
    # Do 1/N weighting
    #
    # Select max_assets_in_portfolio assets in which we are going to invest
    # Calculate a weight for ecah asset in the portfolio using 1/N method based on the raw signal
    alpha_model.select_top_signals(parameters.max_assets_in_portfolio)
    alpha_model.assign_weights(method=weight_by_1_slash_n)

    #
    # Normalise weights and cap the positions
    #

    size_risk_model = USDTVLSizeRiskModel(
        pricing_model=input.pricing_model,
        per_position_cap=parameters.per_position_cap_of_pool,  # This is how much % by all pool TVL we can allocate for a position
    )

    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risk_model,
    )

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(
        state.portfolio,
        ignore_credit=True,
    )
    alpha_model.calculate_target_positions(position_manager)

    # Some debug logging
    if state.backtest_data.ready_at:
        position_manager.log(f"Strategy ready")
    else:
        position_manager.log(f"No pairs with good momentu signal found yet")

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    rebalance_threshold_usd = Parameters.minimum_rebalance_trade_threshold_usd
    assert rebalance_threshold_usd > 150.00
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=rebalance_threshold_usd,  # Don't bother with trades under XXXX USD
        individual_rebalance_min_threshold=10.0,
    )

    #
    # Supply exceed cash as Aave credit
    #
    if parameters.use_credit:
        if len(trades) > 0:
            # We are going to do rebalancing trades and cash in hand is needed.
            # We simplify credit handling by always fully closing the credit position,
            # and then opening it in the next cycle if there are no volatile position rebalances.
            # Opening and closing Aave positions do not have a fee.
            if position_manager.is_any_credit_supply_position_open():
                current_credit_pos = position_manager.get_current_credit_supply_position()
                trades += position_manager.close_credit_supply_position(current_credit_pos)
        else:
            # Check if we can push any extra cash to Aave,
            # As this cycle did not do any rebalancing trades,
            # we should be able to push all exceed cash in hand to Aave.
            cash_to_deposit = position_manager.get_current_cash() * Parameters.allocation
            trades += position_manager.add_cash_to_credit_supply(
                cash_to_deposit,
                min_usd_threshold=Parameters.aave_min_deposit_threshold_usd,
            )

    # Record alpha model state so we can later visualise our alpha model thinking better
    state.visualisation.add_calculations(timestamp, alpha_model.to_dict())

    return trades  # Return the list of trades we made in this cycle
