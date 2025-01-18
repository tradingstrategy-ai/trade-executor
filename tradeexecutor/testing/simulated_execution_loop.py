"""Set up execution loop that connects to an Ethereum testing backend.

"""
import datetime
import queue

from eth_account.signers.local import LocalAccount
from eth_defi.hotwallet import HotWallet
from web3 import Web3

from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import EthereumHotWalletReserveSyncer, HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.state import State
from tradeexecutor.state.store import NoneStore
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol
from tradeexecutor.strategy.universe_model import StaticUniverseModel, StrategyExecutionUniverse
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.timer import timed_task

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2Execution
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import UniswapV2PoolRevaluator

from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3Execution
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import UniswapV3PoolRevaluator

from tradeexecutor.ethereum.one_delta.one_delta_execution import OneDeltaExecution
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.ethereum.one_delta.one_delta_valuation import OneDeltaPoolRevaluator


def set_up_simulated_execution_loop_uniswap_v2(
        *ignore,
        web3: Web3,
        decide_trades: DecideTradesProtocol,
        universe: StrategyExecutionUniverse,
        routing_model: UniswapV2Routing,
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
    assert isinstance(routing_model, UniswapV2Routing)

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

    execution_model = UniswapV2Execution(
        tx_builder,
        max_slippage=1.00,
        confirmation_block_count=0,  # Must be zero for the test chain
    )

    # Pricing model factory for single Uni v2 exchange
    def pricing_model_factory(execution_model, universe, routing_model):
        return UniswapV2LivePricing(
            web3,
            universe.data_universe.pairs,
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
        trade_settle_wait=datetime.timedelta(seconds=1),
        unit_testing=True,
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
    routing_model: UniswapV3Routing,
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
    assert isinstance(routing_model, UniswapV3Routing)

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

    execution_model = UniswapV3Execution(
        tx_builder,
        max_slippage=1.00,
        confirmation_block_count=0,  # Must be zero for the test chain
    )

    # Pricing model factory for single Uni v2 exchange
    def pricing_model_factory(execution_model, universe: StrategyExecutionUniverse, routing_model):
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
        trade_settle_wait=datetime.timedelta(seconds=1),
        unit_testing=True,
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


def set_up_simulated_execution_loop_one_delta(
    *,
    web3: Web3,
    decide_trades: DecideTradesProtocol,
    universe: StrategyExecutionUniverse,
    routing_model: OneDeltaRouting,
    state: State,
    wallet_account = None,
) -> ExecutionLoop:
    """Set up a simulated execution loop for 1delta.

    Create a strategy execution that connects to in-memory blockchain simulation.

    This allows us to step through trades block by block and have
    strategies to respodn to price action (e.g. stop loss)

    See `test_one_delta_live_short.py` for an example.

    :return:
        Execution loop you can manually poke forward tick by tick,
        block by block.
    """
    # assert isinstance(wallet_account, LocalAccount)
    assert isinstance(routing_model, OneDeltaRouting)

    execution_context = ExecutionContext(
        mode=ExecutionMode.simulated_trading,
        engine_version="0.3",
    )

    # Create empty state for this backtest
    store = NoneStore(state)

    hot_wallet = HotWallet(wallet_account)

    # hot_wallet_sync = EthereumHotWalletReserveSyncer(web3, wallet_account.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    cycle_duration = CycleDuration.cycle_unknown

    assert isinstance(universe, TradingStrategyUniverse)

    universe_model = StaticUniverseModel(universe)

    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

    execution_model = OneDeltaExecution(
        tx_builder,
        max_slippage=1.00,
        mainnet_fork=True,
        confirmation_block_count=0,
    )

    def pricing_model_factory(execution_model, universe: StrategyExecutionUniverse, routing_model):
        return OneDeltaLivePricing(
            web3,
            universe.universe.pairs,
            routing_model
        )

    def valuation_model_factory(pricing_model):
        return OneDeltaPoolRevaluator(pricing_model)

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
        unit_testing=True,
        trade_settle_wait=datetime.timedelta(seconds=1),
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

    loop.init_simulation(universe_model, runner)

    return loop


def set_up_simulated_ethereum_generic_execution(
    *,
    web3: Web3,
    decide_trades: DecideTradesProtocol,
    universe: StrategyExecutionUniverse,
    routing_model: GenericRouting,
    pricing_model: GenericPricing,
    valuation_model: GenericValuation,
    state: State,
    hot_wallet: HotWallet,
) -> ExecutionLoop:
    """Set up a simulated execution loop for generic routing.
    """
    assert isinstance(routing_model, GenericRouting)
    assert isinstance(pricing_model, GenericPricing)
    assert isinstance(hot_wallet, HotWallet)

    execution_context = ExecutionContext(
        mode=ExecutionMode.simulated_trading,
        engine_version="0.3",
    )

    # Create empty state for this backtest
    store = NoneStore(state)

    # hot_wallet_sync = EthereumHotWalletReserveSyncer(web3, wallet_account.address)
    sync_model = HotWalletSyncModel(web3, hot_wallet)

    cycle_duration = CycleDuration.cycle_unknown

    assert isinstance(universe, TradingStrategyUniverse)

    universe_model = StaticUniverseModel(universe)

    tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)

    execution_model = EthereumExecution(
        tx_builder,
        max_slippage=1.00,
        mainnet_fork=True,
        confirmation_block_count=0,
    )

    def pricing_model_factory(execution_model, universe: StrategyExecutionUniverse, routing_model):
        return pricing_model

    def valuation_model_factory(pricing_model):
        return valuation_model

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
        unit_testing=True,
        run_state=RunState(),
        trade_settle_wait=datetime.timedelta(seconds=1),
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

    loop.init_simulation(universe_model, runner)

    return loop
