"""Map trades to different routing backends based on their type."""

from typing import Dict, TypeAlias, List, Tuple
import logging

from web3 import Web3

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)

#: Router name to router implementation mapping
#:
#: Router name is like "aave-3", "uniswap-v2" and so on
#:
#:
RoutingMap: TypeAlias = Dict[str, RoutingModel]


class GenericRoutingState(RoutingState):
    """Store one state per router.

    Strategy universe is part of the state, as it will change between
    trading strategy cycles (pairs are added and removed).
    """

    def __init__(
        self,
        strategy_universe: TradingStrategyUniverse,
        state_map: Dict[str, RoutingState]
    ):

        assert isinstance(strategy_universe, TradingStrategyUniverse)
        assert strategy_universe.data_universe.exchange_universe is not None,  "pair_universe is not initialised with exchange data"

        self.strategy_universe = strategy_universe
        self.state_map = state_map


class GenericRouting(RoutingModel):
    """Routes trade to a different routing backend depending on the trade type.

    Based on the pair and exchange data (contained within pairs),
    make choices which router should take the trade.

    Usually there is very direct map exchange -> router,
    as long as the router has the correct smart contract address
    information available for various contracts needed
    to perform the trade.
    """

    def __init__(
        self,
        pair_configurator: PairConfigurator | None,
    ):
        self.pair_configurator = pair_configurator
        logger.info("Initialised %s", self)

    def is_initialised(self) -> bool:
        return self.pair_configurator is not None

    def initialise(self, pair_configurator: PairConfigurator):
        """On a live trading loop GenericRouting gets instiated early, but does not yet have trading universe available.

        This methods allows us to initialise after we have loaded the universe data for the first time.
        """
        self.pair_configurator = pair_configurator

    def create_routing_state(
        self,
        universe: TradingStrategyUniverse,
        execution_details: object
    ) -> RoutingState:
        """Create a new routing state for this cycle.

        :param execution_details:
            A dict of whatever connects live execution to routing.
        """
        assert isinstance(universe, TradingStrategyUniverse)
        substates = {}
        for routing_id in self.pair_configurator.get_supported_routers():
            router = self.pair_configurator.get_config(routing_id).routing_model
            substates[routing_id.router_name] = router.create_routing_state(universe, execution_details)

        return GenericRoutingState(universe, substates)

    def setup_trades(
        self,
        state: State,
        routing_state: GenericRoutingState,
        trades: List[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Route trades.

        - The trade order must be preserved (sells must come before buys,
          so that we have cash)

        - Fills in Blockchain transaction details for each trade
          by using the corresponding router
        """

        assert isinstance(state, State)
        assert isinstance(routing_state, GenericRoutingState)

        for t in trades:
            routing_id = self.pair_configurator.match_router(t.pair)
            protocol_config = self.pair_configurator.get_config(routing_id)

            router = protocol_config.routing_model
            # Set the router, so we know
            # in the post-trade analysis which route this trade took
            t.route = protocol_config.routing_id.router_name

            router_state = routing_state.state_map[protocol_config.routing_id.router_name]
            router.setup_trades(
                state=state,
                routing_state=router_state,
                trades=[t],
                check_balances=check_balances,
                rebroadcast=rebroadcast,
            )

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):
        assert isinstance(state, State)
        assert isinstance(trade, TradeExecution)
        assert type(receipts) == dict
        assert trade.route is not None, f"TradeExecution lacks TradeExecution.route, it was not executed with GenericRouter?\n{t}"

        router = self.pair_configurator.get_routing(trade.pair)

        return router.settle_trade(
            web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure,
        )
