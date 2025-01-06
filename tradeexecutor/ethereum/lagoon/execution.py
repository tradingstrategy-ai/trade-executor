"""Lagoon execution with raw onchain swaps intents."""

import datetime
import logging

from eth_defi.lagoon.vault import LagoonVault
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting, VelvetEnsoRoutingState
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import RoutingStateDetails
from tradeexecutor.strategy.generic.generic_router import GenericRouting, GenericRoutingState
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId

logger = logging.getLogger(__name__)


class LagoonExecution(EthereumExecution):
    """Execution model that uses raw onchain Uniswap swaps with Lagoon."""

    def __init__(self, vault: LagoonVault, **kwargs):
        assert isinstance(vault, LagoonVault)
        super().__init__(**kwargs)
        self.vault = vault

    @staticmethod
    def pre_execute_assertions(
        ts: datetime.datetime,
        routing_model,
        routing_state,
    ):
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_model, VelvetEnsoRouting)
        assert isinstance(routing_state, RoutingState)

    def get_routing_state_details(self) -> RoutingStateDetails:
        details = super().get_routing_state_details()
        details["vault"] = self.vault
        return details

    def create_default_routing_model(
        self,
        strategy_universe: TradingStrategyUniverse,
    ) -> GenericRouting:
        pair_configurator = EthereumPairConfigurator(
            self.web3,
            strategy_universe,
        )
        return GenericRouting(pair_configurator)

    def execute_trades(
        self,
        ts: datetime.datetime,
        state: State,
        trades: list[TradeExecution],
        routing_model: RoutingModel,
        routing_state: RoutingState,
        check_balances=False,
        rebroadcast=False,
        triggered=False,
    ):
        assert isinstance(routing_model, GenericRouting), f"Got {routing_model}"
        assert isinstance(routing_state, GenericRoutingState), f"Got {routing_state}"
        return super().execute_trades(
            ts=ts,
            state=state,
            trades=trades,
            routing_model=routing_model,
            routing_state=routing_state,
            check_balances=check_balances,
            rebroadcast=rebroadcast,
            triggered=triggered,
        )
