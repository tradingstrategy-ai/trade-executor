"""Test default routing options."""
import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.routing_data import get_routing_model, MismatchReserveCurrency
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.reserve_currency import ReserveCurrency


@pytest.fixture()
def execution_context() -> ExecutionContext:
    return ExecutionContext(ExecutionMode.unit_testing_trading)


def test_route_pancakeswap_busd(execution_context):
    """Test Pancake BUSD routing."""
    routing = get_routing_model(execution_context, TradeRouting.pancakeswap_busd, ReserveCurrency.busd)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.bsc

def test_route_pancakeswap_usdc(execution_context):
    """Test Pancake USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.pancakeswap_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.bsc

def test_route_pancakeswap_usdt(execution_context):
    """Test Pancake USDT routing."""
    routing = get_routing_model(execution_context, TradeRouting.pancakeswap_usdt, ReserveCurrency.usdt)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.bsc

def test_route_quickswap_usdt(execution_context):
    """Test Quickswap USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.quickswap_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.polygon

def test_route_quickswap_usdt(execution_context):
    """Test Quickswap USDT routing."""
    routing = get_routing_model(execution_context, TradeRouting.quickswap_usdt, ReserveCurrency.usdt)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.polygon

def test_route_quickswap_dai(execution_context):
    """Test Quickswap DAI routing."""
    routing = get_routing_model(execution_context, TradeRouting.quickswap_dai, ReserveCurrency.dai)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.polygon

def test_route_trader_joe_usdc(execution_context):
    """Test Trader Joe USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.trader_joe_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.avalanche

def test_route_trader_joe_usdt(execution_context):
    """Test Trader Joe USDT routing."""
    routing = get_routing_model(execution_context, TradeRouting.trader_joe_usdt, ReserveCurrency.usdt)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.avalanche

def test_route_ethereum_usdc(execution_context):
    """Test Uniswap v2 USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v2_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.ethereum

def test_route_ethereum_usdt(execution_context):
    """Test Uniswap v2 USDT routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v2_usdt, ReserveCurrency.usdt)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.ethereum

def test_route_ethereum_dai(execution_context):
    """Test Uniswap v2 DAI routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v2_dai, ReserveCurrency.dai)
    assert isinstance(routing, UniswapV2Routing)
    assert routing.chain_id == ChainId.ethereum

def test_route_uniswap_v3_usdc(execution_context):
    """Test Uniswap v3 USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV3Routing)
    assert routing.chain_id == ChainId.ethereum

def test_route_polygon_usdc(execution_context):
    """Test Uniswap v3 Polygon USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc_poly, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV3Routing)
    assert routing.chain_id == ChainId.polygon

def test_route_arbitrum_usdc(execution_context):
    """Test Uniswap v3 Arbitrum USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc_arbitrum_native, ReserveCurrency.usdc)
    assert isinstance(routing, UniswapV3Routing)
    assert routing.chain_id == ChainId.arbitrum

def test_route_arbitrum_usdc(execution_context):
    """Test Uniswap v3 Arbitrum USDC routing."""
    routing = get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc_arbitrum_bridged, ReserveCurrency.usdc_e)
    assert isinstance(routing, UniswapV3Routing)
    assert routing.chain_id == ChainId.arbitrum


def test_route_mismatch_reserve_currency_pancake(execution_context):
    """Test Pancake BUSD routing. """
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.pancakeswap_busd, ReserveCurrency.usdc)

def test_route_mismatch_reserve_currency_quickswap(execution_context):
    """Test Quickswap USDC routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.quickswap_usdc, ReserveCurrency.usdt)

def test_route_mismatch_reserve_currency_trader_joe(execution_context):
    """Test Trader Joe USDC routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.trader_joe_usdc, ReserveCurrency.usdt)

def test_route_mismatch_reserve_currency_uniswap_v3_ethereum(execution_context):
    """Test Uniswap V3 USDC routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc, ReserveCurrency.usdt)

def test_route_mismatch_reserve_currency_uniswap_v3_poly(execution_context):
    """Test Uniswap V3 Polygon USDC routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc_poly, ReserveCurrency.usdt)

def test_route_mismatch_reserve_currency_uniswap_v3_arbitrum(execution_context):
    """Test Uniswap V3 Arbitrum USDC routing."""
    with pytest.raises(MismatchReserveCurrency):
        get_routing_model(execution_context, TradeRouting.uniswap_v3_usdc_arbitrum_bridged, ReserveCurrency.usdt)



