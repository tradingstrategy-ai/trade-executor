"""Test default routing options."""
import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.routing_data import get_routing_model, MismatchReserveCurrency
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.reserve_currency import ReserveCurrency


@pytest.fixture()
def execution_context() -> ExecutionContext:
    return ExecutionContext(ExecutionMode.unit_testing_trading)


def test_route_pancakeswap_busd(execution_context):
    """Test Pancake BUSD routing."""
    routing = get_routing_model(execution_context, TradeRouting.pancakeswap_busd, ReserveCurrency.busd)
    assert isinstance(routing, UniswapV2SimpleRoutingModel)
    assert routing.chain_id == ChainId.bsc


def test_route_mismatch_reserve_currency(execution_context):
    """Test Pancake BUSD routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.pancakeswap_busd, ReserveCurrency.usdc)


def test_route_pancakeswap_usdc(execution_context):
    """Test Pancake USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.pancakeswap_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2SimpleRoutingModel)


def test_route_ethereum_usdc(execution_context):
    """Test Uniswap v2 USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v2_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2SimpleRoutingModel)
    assert routing.chain_id == ChainId.ethereum

