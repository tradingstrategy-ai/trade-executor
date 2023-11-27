"""Map trades to different routing backends based on their type."""

from typing import Collection, Dict, TypeAlias, Protocol, List

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.generic.routing_function import UnroutableTrade
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse

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

        strategy_universe = state.strategy_universe

        for t in trades:
            router_name = self.routing_function(strategy_universe.data_universe.pairs, t)
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



