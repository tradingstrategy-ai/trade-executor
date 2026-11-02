"""Constructed vault fee strategy: deposit, top up, trim and exit through normal APIs."""

from pathlib import Path
from tradeexecutor.testing.vault_fee_sample import load_fee_sample
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

trading_strategy_engine_version = "0.5"


class Parameters:
    """Ordinary strategy parameters; the test supplies its data and evaluation dates."""

    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.user_supplied_routing_model
    initial_cash = 10_000
    backtest_start = None
    backtest_end = None


def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet,
                      strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext) -> None:
    """Register no indicators because this test uses a fixed trading schedule.

    All arguments are unused but retained for the engine's indicator callback
    interface, allowing the fixture strategy to load through the normal runner.
    """


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Exercise acquisition basis and partial/full fee deductions through normal trades.

    Deposit on cycle 1, top up on 10, halve the position on 20 and exit on 30.
    Skip actions while settlement is pending so each reduction owns a distinct
    request.

    :param input:
        Current cycle, state and single-vault fixture universe supplied by the engine.
    :return:
        Ordinary position-manager trades; fees are calculated by the backtester.
    """
    manager = input.get_position_manager()
    pair = next(input.strategy_universe.iterate_pairs())
    if input.cycle == 1:
        return manager.open_spot(pair, value=1000)
    position = next(iter(input.state.portfolio.open_positions.values()), None)
    if position is None or position.has_pending_vault_settlement():
        return []
    if input.cycle == 10:
        return manager.adjust_position(pair, 500, 0, 1.0)
    if input.cycle == 20:
        return manager.adjust_position(pair, -position.get_value() / 2, -position.get_quantity() / 2, 0.5)
    if input.cycle == 30:
        return manager.close_spot_position(position)
    return []


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Load the bundled Lagoon example for standalone module execution.

    Tests supply other sample universes directly. The callback's ``input`` is
    unused because this fixture reads fixed local data.

    :return:
        Single-vault universe built from captured Lagoon metadata and prices.
    """
    return load_fee_sample(Path(__file__).resolve().parents[2] / 'tests/backtest/fixtures/vault_externalised_fees', 'lagoon')
