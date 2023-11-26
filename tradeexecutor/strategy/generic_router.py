"""Map trades to different routing backends based on their type."""

from typing import Collection, Dict, TypeAlias, Protocol

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


#: Router name to router implementation mapping
#:
#: Router name is like "aave-3", "uniswap-v2" and so on
#:
#:
RoutingMap: TypeAlias = Dict[str, RoutingModel]

class RoutingFunction(Protocol):
    """A function protocol definition for router choose.

    """

    def __call__(self, trade: TradeExecution) -> str | None:
        """For each trade, return the name of the route that should be used.

        :return:
            The route name that should be taken.

            If we do not know how to route the trade, return ``None``.
        """


def default_choose_route(trade: TradeExecution) -> str | None:
    """Default router function.

    Use smart contract addresses hardcoded in :py:mod:`tradeexecutor.ethereum.routing_data`.
    """
    return None


class GenericRoutingState(RoutingState):
    """Store one state per router."""

    def __init__(self, state_map: Dict[str, RoutingState]):
        self.state_map = state_map


class GenericRouting(RoutingModel):
    """Routes trade to a different routing backend depending on the trade type."""

    def __init__(self, routers: RoutingMap, routing_function: RoutingFunction = default_choose_route):
        self.routers = routers
        self.routing_function = default_choose_route

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

