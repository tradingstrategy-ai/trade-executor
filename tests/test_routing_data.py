"""Adds more test coverage for routing_data.py"""

import pytest

from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing
from tradeexecutor.ethereum.routing_data import get_routing_model

from tradingstrategy.chain import ChainId


EXECUTION_CONTEXT = ExecutionContext(ExecutionMode.real_trading)

QUICKSWAP_FEE = 0.0030
TRADER_JOE_FEE = 0.0030
PANCAKE_FEE = 0.0025
UNISWAP_V2_FEE = 0.0030

PANCAKE_FACTORY_ROUTER_MAP = {
    "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73": (
        "0x10ED43C718714eb63d5aA57B78B54704E256024E",
        "0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5",
    )
}

QUICKSWAP_FACTORY_ROUTER_MAP = {
    "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32": (
        "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
        "0x96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f",
    )
}

TRADER_JOE_FACTORY_ROUTER_MAP = {
    "0x9Ad6C38BE94206cA50bb0d90783181662f0Cfa10": (
        "0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
        "0x0bbca9af0511ad1a1da383135cf3a8d2ac620e549ef9f6ae3a4c33c2fed0af91",
    )
}

UNISWAP_V2_FACTORY_MAP = {
    "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f": (
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
        "96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f",
    )
}

UNISWAP_V3_ADDRESS_MAP = {
    "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
    "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    "position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
    "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
    # "router02":"0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    # "quoterV2":"0x61fFE014bA17989E743c5F6cB21bF9697530B21e"
}


@pytest.mark.parametrize(
    "trade_routing,reserve_currency,allowed_intermediary_pairs,reserve_token_address",
    [
        (
            TradeRouting.pancakeswap_busd,
            ReserveCurrency.busd,
            {
                "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0x58f876857a02d6762e0101bb5c46a8c1ed44dc16",
            },
            "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56".lower(),
        ),
        (
            TradeRouting.pancakeswap_usdc,
            ReserveCurrency.usdc,
            {
                "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0xd99c7f6c65857ac913a8f880a4cb84032ab2fc5b",
            },
            "0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d".lower(),
        ),
        (
            TradeRouting.pancakeswap_usdt,
            ReserveCurrency.usdt,
            {
                "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c": "0x16b9a82891338f9ba80e2d6970fdda79d1eb0dae",
            },
            "0x55d398326f99059ff775485246999027b3197955".lower(),
        ),
    ],
)
def test_pancake_routing_models(
    trade_routing, reserve_currency, allowed_intermediary_pairs, reserve_token_address
):
    routing_model = get_routing_model(
        EXECUTION_CONTEXT, trade_routing, reserve_currency
    )

    expected_model = UniswapV2Routing(
        PANCAKE_FACTORY_ROUTER_MAP,
        allowed_intermediary_pairs,
        reserve_token_address,
        ChainId.bsc,
        PANCAKE_FEE,
    )

    assert (
        routing_model.allowed_intermediary_pairs
        == expected_model.allowed_intermediary_pairs
    )
    assert routing_model.factory_router_map == expected_model.factory_router_map
    assert routing_model.reserve_token_address == expected_model.reserve_token_address
    assert routing_model.chain_id == expected_model.chain_id
    assert routing_model.trading_fee == expected_model.trading_fee


@pytest.mark.parametrize(
    "trade_routing,reserve_currency,allowed_intermediary_pairs,reserve_token_address",
    [
        (
            TradeRouting.quickswap_usdc,
            ReserveCurrency.usdc,
            {
                "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827",
                "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "0x853ee4b2a13f8a742d64c8f088be7ba2131f670d",
            },
            "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174".lower(),
        ),
        (
            TradeRouting.quickswap_usdt,
            ReserveCurrency.usdt,
            {
                "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0x604229c960e5cacf2aaeac8be68ac07ba9df81c3",
            },
            "0xc2132d05d31c914a87c6611c10748aeb04b58e8f".lower(),
        ),
        (
            TradeRouting.quickswap_dai,
            ReserveCurrency.dai,
            {
                "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "0xf04adbf75cdfc5ed26eea4bbbb991db002036bdd"
            },
            "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063".lower(),
        ),
    ],
)
def test_quickswap_routing_models(
    trade_routing, reserve_currency, allowed_intermediary_pairs, reserve_token_address
):
    routing_model = get_routing_model(
        EXECUTION_CONTEXT, trade_routing, reserve_currency
    )

    expected_model = UniswapV2Routing(
        QUICKSWAP_FACTORY_ROUTER_MAP,
        allowed_intermediary_pairs,
        reserve_token_address,
        ChainId.polygon,
        QUICKSWAP_FEE,
    )

    assert (
        routing_model.allowed_intermediary_pairs
        == expected_model.allowed_intermediary_pairs
    )
    assert routing_model.factory_router_map == expected_model.factory_router_map
    assert routing_model.reserve_token_address == expected_model.reserve_token_address
    assert routing_model.chain_id == expected_model.chain_id
    assert routing_model.trading_fee == expected_model.trading_fee


@pytest.mark.parametrize(
    "trade_routing,reserve_currency,allowed_intermediary_pairs,reserve_token_address",
    [
        (
            TradeRouting.trader_joe_usdc,
            ReserveCurrency.usdc,
            {
                "0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7": "0xf4003f4efbe8691b60249e6afbd307abe7758adb",
            },
            "0xb97ef9ef8734c71904d8002f8b6bc66dd9c48a6e".lower(),
        ),
        (
            TradeRouting.trader_joe_usdt,
            ReserveCurrency.usdt,
            {
                "0xb31f66aa3c1e785363f0875a1b74e27b85fd66c7": "0xbb4646a764358ee93c2a9c4a147d5aded527ab73",
            },
            "0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7".lower(),
        ),
    ],
)
def test_trader_joe_routing_models(
    trade_routing, reserve_currency, allowed_intermediary_pairs, reserve_token_address
):
    routing_model = get_routing_model(
        EXECUTION_CONTEXT, trade_routing, reserve_currency
    )

    expected_model = UniswapV2Routing(
        TRADER_JOE_FACTORY_ROUTER_MAP,
        allowed_intermediary_pairs,
        reserve_token_address,
        ChainId.avalanche,
        TRADER_JOE_FEE,
    )

    assert (
        routing_model.allowed_intermediary_pairs
        == expected_model.allowed_intermediary_pairs
    )
    assert routing_model.factory_router_map == expected_model.factory_router_map
    assert routing_model.reserve_token_address == expected_model.reserve_token_address
    assert routing_model.chain_id == expected_model.chain_id
    assert routing_model.trading_fee == expected_model.trading_fee


@pytest.mark.parametrize(
    "trade_routing,reserve_currency,allowed_intermediary_pairs,reserve_token_address",
    [
        (
            TradeRouting.uniswap_v2_usdc,
            ReserveCurrency.usdc,
            {
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc",
            },
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48".lower(),
        ),
        (
            TradeRouting.uniswap_v2_usdt,
            ReserveCurrency.usdt,
            {
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "0x0d4a11d5eeaac28ec3f61d100daf4d40471f1852",
            },
            "0xdac17f958d2ee523a2206206994597c13d831ec7".lower(),
        ),
        (
            TradeRouting.uniswap_v2_dai,
            ReserveCurrency.dai,
            {
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2": "0xa478c2975ab1ea89e8196811f51a7b7ade33eb11"
            },
            "0x6b175474e89094c44da98b954eedeac495271d0f".lower(),
        ),
    ],
)
def test_uniswap_v2_routing_models(
    trade_routing, reserve_currency, allowed_intermediary_pairs, reserve_token_address
):
    routing_model = get_routing_model(
        EXECUTION_CONTEXT, trade_routing, reserve_currency
    )

    expected_model = UniswapV2Routing(
        UNISWAP_V2_FACTORY_MAP,
        allowed_intermediary_pairs,
        reserve_token_address,
        ChainId.ethereum,
        UNISWAP_V2_FEE,
    )

    assert (
        routing_model.allowed_intermediary_pairs
        == expected_model.allowed_intermediary_pairs
    )
    assert routing_model.factory_router_map == expected_model.factory_router_map
    assert routing_model.reserve_token_address == expected_model.reserve_token_address
    assert routing_model.chain_id == expected_model.chain_id
    assert routing_model.trading_fee == expected_model.trading_fee


@pytest.mark.parametrize(
    "trade_routing,reserve_currency,allowed_intermediary_pairs,reserve_token_address, chain_id",
    [
        (
            TradeRouting.uniswap_v3_usdc,
            ReserveCurrency.usdc,
            {
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                "0xdac17f958d2ee523a2206206994597c13d831ec7": "0x3416cf6c708da44db2624d63ea0aaef7113527c6",
            },
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".lower(),
            ChainId.ethereum,
        ),
        (
            TradeRouting.uniswap_v3_usdt,
            ReserveCurrency.usdt,
            {
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "0x11b815efb8f581194ae79006d24e0d814b7697f6",
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0x3416cf6c708da44db2624d63ea0aaef7113527c6",
            },
            "0xdac17f958d2ee523a2206206994597c13d831ec7",
            ChainId.ethereum,
        ),
        (
            TradeRouting.uniswap_v3_dai,
            ReserveCurrency.dai,
            {
                "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0x5777d92f208679db4b9778590fa3cab3ac9e2168",
            },
            "0x6b175474e89094c44da98b954eedeac495271d0f",
            ChainId.ethereum,
        ),
        (
            TradeRouting.uniswap_v3_busd,
            ReserveCurrency.busd,
            {
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "0x5e35c4eba72470ee1177dcb14dddf4d9e6d915f4",
                "0xdac17f958d2ee523a2206206994597c13d831ec7": "0xd5ad5ec825cac700d7deafe3102dc2b6da6d195d",
                "0x6b175474e89094c44da98b954eedeac495271d0f": "0xd1000344c3a00846462b4624bb452621cf2ce001",
            },
            "0x4fabb145d64652a948d72533023f6e7a623c7c53",
            ChainId.ethereum,
        ),
        (
            TradeRouting.uniswap_v3_usdc_poly,
            ReserveCurrency.usdc,
            {
                "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0xa374094527e1673a86de625aa59517c5de346d32",
                "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": "0x45dda9cb7c25131df268515131f647d726f50608",
            },
            "0x2791bca1f2de4661ed88a30c99a7a9449aa84174".lower(),
            ChainId.polygon,
        ),
        (
            TradeRouting.uniswap_v3_usdt_poly,
            ReserveCurrency.usdt,
            {
                "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": "0x9b08288c3be4f62bbf8d1c20ac9c5e6f9467d8b7",
            },
            "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
            ChainId.polygon,
        ),
        (
            TradeRouting.uniswap_v3_usdc_arbitrum_native,
            ReserveCurrency.usdc,
            {},
            "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
            ChainId.arbitrum,
        ),
        (
                TradeRouting.uniswap_v3_usdc_arbitrum_bridged,
                ReserveCurrency.usdc_e,
                {},
                "0xff970a61a04b1ca14834a43f5de4533ebddb5cc8",
                ChainId.arbitrum,
        ),
        (
            TradeRouting.uniswap_v3_usdt_arbitrum,
            ReserveCurrency.usdt,
            {
                "0x82af49447d8a07e3bd95bd0d56f35241523fbab1": "0x641c00a822e8b671738d32a431a4fb6074e5c79d"
            },
            "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
            ChainId.arbitrum,
        ),
    ],
)
def test_uniswap_v3_routing_models(
    trade_routing,
    reserve_currency,
    allowed_intermediary_pairs,
    reserve_token_address,
    chain_id,
):
    routing_model = get_routing_model(
        EXECUTION_CONTEXT, trade_routing, reserve_currency
    )

    expected_model = UniswapV3Routing(
        UNISWAP_V3_ADDRESS_MAP,
        allowed_intermediary_pairs,
        reserve_token_address,
        chain_id,
    )

    assert (
        routing_model.allowed_intermediary_pairs
        == expected_model.allowed_intermediary_pairs
    )
    assert routing_model.address_map == expected_model.address_map
    assert routing_model.reserve_token_address == expected_model.reserve_token_address
    assert routing_model.chain_id == expected_model.chain_id
