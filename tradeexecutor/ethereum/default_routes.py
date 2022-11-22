"""Default routing models for Uniswap v2 like exchange."""
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.routing import RoutingModel


def get_pancake_default_routing_parameters(reserve_currency: ReserveCurrency) -> dict:
    """Generate routing using PancakeSwap v2 router.

    - Trade WBNB and BUSD pairs

    TODO: Polish the interface of this function when we have more strategies

    :param reserve_currency: BUSD accepted
    """
    assert reserve_currency == ReserveCurrency.busd, f"Only BUSD supported for this routing module, received {reserve_currency}"

    #
    # Routing options
    #

    # Keep everything internally in BUSD
    reserve_token_address = "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower()

    # Allowed exchanges as factory -> router pairs,
    # by their smart contract addresses
    factory_router_map = {
        "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": ("0x10ED43C718714eb63d5aA57B78B54704E256024E", "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5")
    }

    # For three way trades, which pools we can use
    allowed_intermediary_pairs = {
        # Route WBNB through BUSD:WBNB pool,
        "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0x58f876857a02d6762e0101bb5c46a8c1ed44dc16",
    }

    return {
        "factory_router_map": factory_router_map,
        "allowed_intermediary_pairs": allowed_intermediary_pairs,
        "reserve_token_address": reserve_token_address,
        "quote_token_addresses": {"0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c", "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"}
    }


def create_pancake_routing(reserve_currency: ReserveCurrency) -> UniswapV2SimpleRoutingModel:

    params = get_pancake_default_routing_parameters(reserve_currency)

    routing_model = UniswapV2SimpleRoutingModel(
        params["factory_router_map"],
        params["allowed_intermediary_pairs"],
        params["reserve_token_address"],
    )

    return routing_model



def get_backtest_routing_model(routing_type: TradeRouting, reserve_currency: ReserveCurrency) -> BacktestRoutingModel:
    """Hardcoded routing model support for backtests."""

    assert isinstance(routing_type, TradeRouting)

    if routing_type == TradeRouting.pancakeswap_basic:
        params = get_pancake_default_routing_parameters(reserve_currency)
        return BacktestRoutingModel(
            params["factory_router_map"],
            params["allowed_intermediary_pairs"],
            params["reserve_token_address"],
        )

    raise NotImplementedError(f"The routing model is not supported: {routing_type.value}")


def get_routing_model(
        execution_context: ExecutionContext,
        routing_type: TradeRouting,
        reserve_currency: ReserveCurrency) -> RoutingModel:
    """Create trade routing model for the strategy.

    :param execution_model:
        Either backtest or live

    :param routing_type:
        One of the default routing options, as definedin backtest notebook or strategy module

    :param reserve_currency:
        One of the default reserve currency options, as definedin backtest notebook or strategy module
    """

    if execution_context.mode == ExecutionMode.backtesting:
        return get_backtest_routing_model(routing_type, reserve_currency)

    if routing_type == TradeRouting.pancakeswap_basic:
        return create_pancake_routing(reserve_currency)

    raise NotImplementedError("Not yet done - update get_routing_model to support this default routing option")