"""Velvet execution with Enso intents."""

import datetime
import logging


from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting, VelvetEnsoRoutingState
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import RoutingStateDetails
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId

logger = logging.getLogger(__name__)


class VelvetExecution(EthereumExecution):
    """Execution model that is paired with Enso intents and Velvet vault."""

    def __init__(self, vault: VelvetVault, **kwargs):
        assert isinstance(vault, VelvetVault)
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
    ) -> VelvetEnsoRouting:
        reserve_asset = strategy_universe.get_reserve_asset()

        # A hardcoded hack for now
        assert self.web3.eth.chain_id == ChainId.base.value
        allowed_intermediary_pairs = {
            "0x4200000000000000000000000000000000000006": "0x88A43bbDF9D098eEC7bCEda4e2494615dfD9bB9C",
        }

        return VelvetEnsoRouting(
            allowed_intermediary_pairs=allowed_intermediary_pairs,
            reserve_token_address=reserve_asset.address,
        )

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
        assert isinstance(routing_model, VelvetEnsoRouting), f"Got {routing_model}"
        assert isinstance(routing_state, VelvetEnsoRoutingState), f"Got {routing_state}"
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
