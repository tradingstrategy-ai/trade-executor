"""1delta live pricing.

Currently subclass Uniswap v3 live pricing.
"""
import logging
import datetime
from decimal import Decimal
from typing import Optional, Dict

from eth_defi.provider.broken_provider import get_block_tip_latency
from web3 import Web3

from tradeexecutor.ethereum.one_delta.one_delta_execution import OneDeltaExecutionModel
from tradeexecutor.ethereum.one_delta.one_delta_routing import OneDeltaRouting
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.eth_pricing_model import EthereumPricingModel, LP_FEE_VALIDATION_EPSILON
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradingstrategy.pair import PandasPairUniverse

from eth_defi.uniswap_v3.price import UniswapV3PriceHelper, estimate_sell_received_amount, estimate_buy_received_amount, get_onchain_price
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment

logger = logging.getLogger(__name__)


class OneDeltaLivePricing(UniswapV3LivePricing):
    """1delta live pricing."""

    def __init__(
        self,
        web3: Web3,
        pair_universe: PandasPairUniverse,
        routing_model: OneDeltaRouting,
        very_small_amount: float = Decimal("0.10"),
        epsilon: float | None = LP_FEE_VALIDATION_EPSILON,
    ):
        assert isinstance(routing_model, OneDeltaRouting)

        super().__init__(
            web3,
            pair_universe,
            routing_model,
            very_small_amount,
            epsilon
        )


def one_delta_live_pricing_factory(
    execution_model: ExecutionModel,
    universe: TradingStrategyUniverse,
    routing_model: OneDeltaRouting,
) -> OneDeltaLivePricing:
    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, OneDeltaExecutionModel), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, OneDeltaRouting), f"This pricing method only works with 1delta routing model, we received {routing_model}"

    return OneDeltaLivePricing(
        execution_model.web3,
        universe.data_universe.pairs,
        routing_model,
    )
