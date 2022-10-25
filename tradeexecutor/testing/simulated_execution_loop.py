"""Set up execution loop that connects to an Ethereum testing backend.

"""
import datetime
import queue

from eth_account.signers.local import LocalAccount
from eth_defi.hotwallet import HotWallet
from web3 import Web3

from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v2_valuation import UniswapV2PoolRevaluator
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol
from tradeexecutor.strategy.universe_model import StaticUniverseModel, StrategyExecutionUniverse
from tradeexecutor.utils.timer import timed_task


def set_up_simulated_execution_loop_uniswap_v2(
        *ignore,
        web3: Web3,
        decide_trades: DecideTradesProtocol,
        universe: StrategyExecutionUniverse,
        routing_model: UniswapV2SimpleRoutingModel,
        state: State,
        wallet_account: LocalAccount,
) -> ExecutionLoop:
    """Set up a simulated execution loop.

    Create a strategy execution that connects to in-memory blockchain simulation.

    This allows us to step through trades block by block and have
    strategies to respodn to price action (e.g. stop loss)

    Currently hardcoded for Uniswap v2 exchanges only.

    See `test_uniswap_live_stop_loss.py` for an example.

    :return:
        Execution loop you can manually poke forward tick by tick,
        block by block.
    """

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    assert isinstance(wallet_account, LocalAccount)
    assert isinstance(routing_model, UniswapV2SimpleRoutingModel)

    execution_context = ExecutionContext(
        mode=ExecutionMode.simulated_trading,
    )

    # Create empty state for this backtest
    store = NoneStore(state)

    hot_wallet = HotWallet(wallet_account)

    hot_wallet_sync = EthereumHotWalletReserveSyncer(web3, wallet_account.address)

    cycle_duration = CycleDuration.cycle_unknown

    universe_model = StaticUniverseModel(universe)

    execution_model = UniswapV2ExecutionModel(
        web3,
        hot_wallet,
        max_slippage=1.00,
    )

    # Pricing model factory for single Uni v2 exchange
    def pricing_model_factory(execution_model, universe, routing_model):
        return UniswapV2LivePricing(
            web3,
            universe.universe.pairs,
            routing_model)

    # Valuation model factory for single Uni v2 exchange
    def valuation_model_factory(pricing_model):
        return UniswapV2PoolRevaluator(pricing_model)

    runner = PandasTraderRunner(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=UncheckedApprovalModel(),
        valuation_model_factory=valuation_model_factory,
        sync_method=hot_wallet_sync,
        pricing_model_factory=pricing_model_factory,
        routing_model=routing_model,
        decide_trades=decide_trades,
    )

    loop = ExecutionLoop(
        name="simulated_execution",
        command_queue=queue.Queue(),
        execution_context=execution_context,
        execution_model=execution_model,
        sync_method=hot_wallet_sync,
        pricing_model_factory=pricing_model_factory,
        valuation_model_factory=valuation_model_factory,
        strategy_factory=None,
        store=store,
        cycle_duration=cycle_duration,
        client=None,
        approval_model=UncheckedApprovalModel(),
        stats_refresh_frequency=datetime.timedelta(0),
        position_trigger_check_frequency=datetime.timedelta(0),
    )

    loop.init_simulation(
        universe_model,
        runner,
    )

    return loop
