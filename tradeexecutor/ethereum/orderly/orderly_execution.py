"""Orderly execution with raw onchain swaps intents."""

import datetime
import logging

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import RoutingStateDetails
from tradeexecutor.ethereum.orderly.orderly_routing import OrderlyRouting, OrderlyRoutingState
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


class OrderlyExecution(EthereumExecution):
    """Execution model that uses Orderly vault for deposits and withdrawals."""

    def __init__(
        self,
        vault: OrderlyVault,
        broker_id: str,
        orderly_account_id: str,
        **kwargs
    ):
        assert isinstance(vault, OrderlyVault)
        super().__init__(**kwargs)
        self.vault = vault
        self.broker_id = broker_id
        self.orderly_account_id = orderly_account_id

    @staticmethod
    def pre_execute_assertions(
        ts: datetime.datetime,
        routing_model,
        routing_state,
    ):
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_state, RoutingState)

    def check_valid(self):
        """Check that the execution model is valid.

        - All smart contracts are properly configured

        :raise ASsertionError:
            Smart contracts not properly configured or not enabled.
        """
        # Check that the Orderly vault is properly configured
        assert self.vault.contract is not None, f"Orderly vault contract not initialized for {self.vault}"
        assert self.vault.address is not None, f"Orderly vault address not set for {self.vault}"

    def get_routing_state_details(self) -> RoutingStateDetails:
        details = super().get_routing_state_details()
        details["vault"] = self.vault
        details["broker_id"] = self.broker_id
        details["orderly_account_id"] = self.orderly_account_id
        return details

    def create_default_routing_model(
        self,
        strategy_universe: TradingStrategyUniverse,
    ) -> OrderlyRouting:
        # Validate vault configuration before creating routing model
        self.check_valid()

        reserve_asset = strategy_universe.get_reserve_asset()

        return OrderlyRouting(
            reserve_token_address=reserve_asset.address,
            vault=self.vault,
            broker_id=self.broker_id,
            orderly_account_id=self.orderly_account_id,
            token_id_mapping=self.vault.supported_tokens,
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
        assert isinstance(routing_model, OrderlyRouting), f"Got {routing_model}"
        assert isinstance(routing_state, OrderlyRoutingState), f"Got {routing_state}"
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