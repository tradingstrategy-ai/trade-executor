"""Map trades to different routing backends based on their type."""

from typing import Dict, TypeAlias, List, Tuple

from web3 import Web3

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.generic.routing_function import UnroutableTrade, default_route_chooser, RoutingFunction
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


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
            routers: RoutingMap,
            routing_function: RoutingFunction = default_route_chooser,
    ):
        self.routers = routers
        self.routing_function = routing_function

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
        for router_name, router in self.routers.items():
            substates[router_name] = router.create_routing_state(universe, execution_details)

        return GenericRoutingState(universe, substates)

    def get_router(
        self,
        strategy_universe: TradingStrategyUniverse,
        trade: TradeExecution,
    ) -> Tuple[str, RoutingModel]:
        t = trade
        router_name = self.routing_function(strategy_universe.data_universe.pairs, t.pair)
        if router_name is None:
            raise UnroutableTrade(
                f"Cannot route: {t}\n"
                f"Using routing function: {self.routing_function}"
                f"Available routes: {list(self.routers.keys())}"
            )

        router = self.routers.get(router_name)
        if router is None:
            raise UnroutableTrade(
                f"Router not available: {router_name} for {t}\n"
                f"Trade routing function give us a route: {router_name}, but it is not configured\n"
                f"Available routes: {list(self.routers.keys())}"
            )

        return router_name, router

    def setup_trades(
            self,
            state: GenericRoutingState,
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

        assert isinstance(state, GenericRoutingState)

        strategy_universe = state.strategy_universe

        for t in trades:
            router_name, router = self.get_router(strategy_universe, t)

            # Set the router, so we know
            # in the post-trade analysis which route this trade took
            t.route = router_name

            router_state = state.state_map[router_name]
            router.setup_trades(
                router_state,
                [t],
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
        router = self.routers[trade.route]
        return router.settle_trade(
            web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure,
        )
