"""Set up execution loop that connects to an Ethereum testing backend.

"""
import datetime
import queue

from eth_account.signers.local import LocalAccount
from eth_defi.hotwallet import HotWallet
from web3 import Web3

from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.hot_wallet_sync_model import EthereumHotWalletReserveSyncer, HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol
from tradeexecutor.strategy.universe_model import StaticUniverseModel, StrategyExecutionUniverse
from tradeexecutor.utils.timer import timed_task

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import UniswapV3PoolRevaluator


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

    See `test_uniswap_live_stop_loss.py` for an example.

    :param wallet_account:
        A trader account with some deployed money

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

    # hot_wallet_sync = EthereumHotWalletReserveSyncer(web3, wallet_account.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)
    sync_model.setup_all(state, list(universe.reserve_assets))

    cycle_duration = CycleDuration.cycle_unknown

    universe_model = StaticUniverseModel(universe)

    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

    execution_model = UniswapV2ExecutionModel(
        tx_builder,
        max_slippage=1.00,
        confirmation_block_count=0,  # Must be zero for the test chain
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
        sync_model=sync_model,
        pricing_model_factory=pricing_model_factory,
        routing_model=routing_model,
        decide_trades=decide_trades,
        execution_context=execution_context,
    )

    loop = ExecutionLoop(
        name="simulated_execution",
        command_queue=queue.Queue(),
        execution_context=execution_context,
        execution_model=execution_model,
        sync_model=sync_model,
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


def set_up_simulated_execution_loop_uniswap_v3(
        *ignore,
        web3: Web3,
        decide_trades: DecideTradesProtocol,
        universe: StrategyExecutionUniverse,
        routing_model: UniswapV3SimpleRoutingModel,
        state: State,
        wallet_account: LocalAccount,
) -> ExecutionLoop:
    """Set up a simulated execution loop for Uniswap V3.

    Create a strategy execution that connects to in-memory blockchain simulation.

    This allows us to step through trades block by block and have
    strategies to respodn to price action (e.g. stop loss)

    See `test_uniswap_live_stop_loss_uniswap_v3.py` for an example.

    :return:
        Execution loop you can manually poke forward tick by tick,
        block by block.
    """

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    assert isinstance(wallet_account, LocalAccount)
    assert isinstance(routing_model, UniswapV3SimpleRoutingModel)

    execution_context = ExecutionContext(
        mode=ExecutionMode.simulated_trading,
    )

    # Create empty state for this backtest
    store = NoneStore(state)

    hot_wallet = HotWallet(wallet_account)

    # hot_wallet_sync = EthereumHotWalletReserveSyncer(web3, wallet_account.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    cycle_duration = CycleDuration.cycle_unknown

    universe_model = StaticUniverseModel(universe)

    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

    execution_model = UniswapV3ExecutionModel(
        tx_builder,
        max_slippage=1.00,
        confirmation_block_count=0,  # Must be zero for the test chain
    )

    # Pricing model factory for single Uni v2 exchange
    def pricing_model_factory(execution_model, universe, routing_model):
        return UniswapV3LivePricing(
            web3,
            universe.universe.pairs,
            routing_model)

    # Valuation model factory for single Uni v2 exchange
    def valuation_model_factory(pricing_model):
        return UniswapV3PoolRevaluator(pricing_model)

    runner = PandasTraderRunner(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=UncheckedApprovalModel(),
        valuation_model_factory=valuation_model_factory,
        sync_model=sync_model,
        pricing_model_factory=pricing_model_factory,
        routing_model=routing_model,
        decide_trades=decide_trades,
        execution_context=execution_context,
    )

    loop = ExecutionLoop(
        name="simulated_execution",
        command_queue=queue.Queue(),
        execution_context=execution_context,
        execution_model=execution_model,
        sync_model=sync_model,
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
