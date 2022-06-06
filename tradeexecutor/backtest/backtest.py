from decimal import Decimal

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncer
from tradeexecutor.backtest.backtest_valuation import backtest_valuation_factory
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.strategy.execution_type import TradeExecutionType


def create_trade_execution_model(
        execution_type: TradeExecutionType,
        max_slippage: float,
        start_balance: Decimal,
):

    assert execution_type == execution_type.backtest

    wallet = SimulatedWallet()
    sync_method = BacktestSyncer(wallet, start_balance)
    execution_model = BacktestExecutionModel(max_slippage=max_slippage)
    valuation_model_factory = backtest_valuation_factory
    pricing_model_factory = v
    return execution_model, sync_method, valuation_model_factory, pricing_model_factory
