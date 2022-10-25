"""Set up execution loop that connects to an Ethereum testing backend.

"""
from contextlib import AbstractContextManager

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
from tradeexecutor.state.sync import SyncMethod
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol, CURRENT_ENGINE_VERSION
from tradeexecutor.strategy.universe_model import StaticUniverseModel, StrategyExecutionUniverse
from tradeexecutor.strategy.valuation import ValuationModelFactory


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
    """

    if ignore:
        # https://www.python.org/dev/peps/pep-3102/
        raise TypeError("Only keyword arguments accepted")

    assert isinstance(wallet_account, LocalAccount)

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
        return UniswapV2LivePricing(universe, routing_model)

    # Valuation model factory for single Uni v2 exchange
    def valuation_model_factory(pricing_model):
        return UniswapV2PoolRevaluator(pricing_model)

    # The main factory that sets up a strategy runner for our config
    def strategy_factory(
            *ignore,
            execution_model: ExecutionModel,
            sync_method: SyncMethod,
            pricing_model_factory: PricingModelFactory,
            valuation_model_factory: ValuationModelFactory,
            client: Optional[Client],
            timed_task_context_manager: AbstractContextManager,
            approval_model: ApprovalModel,
            routing_model: Optional[RoutingModel] = None,
    ):
        runner = PandasTraderRunner(
            timed_task_context_manager=timed_task_context_manager,
            execution_model=execution_model,
            approval_model=approval_model,
            valuation_model_factory=valuation_model_factory,
            sync_method=sync_method,
            pricing_model_factory=pricing_model_factory,
            routing_model=routing_model,
            decide_trades=decide_trades,
        )

        description = StrategyExecutionDescription(
            runner=runner,
            universe_model=universe_model,
            trading_strategy_engine_version=CURRENT_ENGINE_VERSION,
            cycle_duration=cycle_duration,
        )
        return description

    loop = ExecutionLoop(
        execution_context=execution_context,
        execution_model=execution_model,
        sync_method=hot_wallet_sync,
        pricing_model_factory=pricing_model_factory,
        valuation_model_factory=valuation_model_factory,
        strategy_factory=strategy_factory,
        store=store,
        cycle_duration=cycle_duration,
        client=None,
        approval_model=UncheckedApprovalModel(),
    )

    return loop
