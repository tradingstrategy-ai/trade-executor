"""Constructed vault fee strategy: deposit, top up, trim and exit through normal APIs."""

from pathlib import Path
from tradeexecutor.testing.vault_fee_sample import load_fee_sample
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput

trading_strategy_engine_version = "0.5"


class Parameters:
    """Ordinary strategy parameters; the test supplies its data and evaluation dates."""

    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.user_supplied_routing_model
    initial_cash = 10_000
    backtest_start = None
    backtest_end = None


def create_indicators(parameters, indicators, strategy_universe, execution_context):
    """No indicators are needed for the constructed schedule."""


def decide_trades(input: StrategyInput):
    """Exercise additions and partial/full reductions without strategy fee handling."""
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


def create_trading_universe(input):
    """Load the bundled Lagoon example; tests supply other real sample universes."""
    return load_fee_sample(Path(__file__).resolve().parents[2] / 'tests/backtest/fixtures/vault_externalised_fees', 'lagoon')
