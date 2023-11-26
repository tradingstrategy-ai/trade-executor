"""Map trades to different routing backends based on their type."""

from typing import Collection, Dict, TypeAlias, Protocol, List

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


#: Router name to router implementation mapping
#:
#: Router name is like "aave-3", "uniswap-v2" and so on
#:
#:
RoutingMap: TypeAlias = Dict[str, RoutingModel]


class UnroutableTrade(Exception):
    """Trade cannot be routed, as we could not find a matching route."""


class RoutingFunction(Protocol):
    """A function protocol definition for router choose.

    """

    def __call__(self, trade: TradeExecution) -> str | None:
        """For each trade, return the name of the route that should be used.

        :return:
            The route name that should be taken.

            If we do not know how to route the trade, return ``None``.
        """


def default_route_chooser(trade: TradeExecution) -> str | None:
    """Default router function.

    Comes with some DEXes and protocols prebuilt.

    Use smart contract addresses hardcoded in :py:mod:`tradeexecutor.ethereum.routing_data`.
    """
    return None


class GenericRoutingState(RoutingState):
    """Store one state per router."""

    def __init__(self, state_map: Dict[str, RoutingState]):
        self.state_map = state_map


class GenericRouting(RoutingModel):
    """Routes trade to a different routing backend depending on the trade type."""

    def __init__(
            self,
            routers: RoutingMap,
            routing_function: RoutingFunction = default_route_chooser
    ):
        self.routers = routers
        self.routing_function = routing_function

    def create_routing_state(
            self,
            universe: StrategyExecutionUniverse,
            execution_details: object
    ) -> RoutingState:
        """Create a new routing state for this cycle.

        :param execution_details:
            A dict of whatever connects live execution to routing.
        """
        substates = {}
        for router_name, router in self.routers.items():
            substates[router_name] = router.create_routing_state(universe, execution_details)

        return GenericRoutingState(substates)

    def setup_trades(
            self,
            state: GenericRoutingState,
            trades: List[TradeExecution],
            check_balances=False
    ):
        """Route trades.

        - The trade order must be preserved (sells must come before buys,
          so that we have cash)

        - Fills in Blockchain transaction details for each trade
          by using the corresponding router
        """

        assert isinstance(state, GenericRoutingState)

        for t in trades:
            router_name = self.routing_function(t)
            if router_name is None:
                raise UnroutableTrade(
                    f"Cannot route: {t}\n"
                    f"Using routing function: {self.routing_function}"
                    f"Available routes: {list(self.routers.keys())}"
                )

            router = self.routers.get(router_name)
            if router is None:
                raise UnroutableTrade(
                    f"Router not available: {t}\n"
                    f"Trade routing function give us a route: {router_name}, but it is not configured\n"
                    f"Available routes: {list(self.routers.keys())}"
                )

            router_state = state.state_map[router_name]
            router.setup_trades(
                router_state,
                [t],
                check_balances=check_balances,
            )



