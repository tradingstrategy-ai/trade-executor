"""Velvet execution with Enso intents."""

import datetime
import logging

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from tradeexecutor.strategy.execution_model import RoutingStateDetails
from tradeexecutor.strategy.routing import RoutingState

logger = logging.getLogger(__name__)

class VelvetExecution(EthereumExecution):
    """Execution model that is paired with Enso intents."""

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
