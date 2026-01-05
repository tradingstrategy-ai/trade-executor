"""Freqtrade routing model for deposit/withdrawal management."""

import logging
from typing import List

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.freqtrade.config import FreqtradeConfig, FreqtradeDepositMethod

logger = logging.getLogger(__name__)


class FreqtradeRoutingModel(RoutingModel):
    """Route capital deposits/withdrawals to Freqtrade instances.

    Phase 1 implementation: Manual mode only.
    - Deposits: User deposits manually, routing just logs instructions
    - Withdrawals: User withdraws manually, routing tracks state
    """

    def __init__(self, freqtrade_configs: dict[str, FreqtradeConfig]):
        """Initialize routing model.

        Args:
            freqtrade_configs: Dict mapping freqtrade_id -> FreqtradeConfig
        """
        super().__init__()
        self.freqtrade_configs = freqtrade_configs

    def setup_trades(
        self,
        state: State,
        trades: List[TradeExecution],
        routing_state: RoutingState,
        **kwargs,
    ):
        """Prepare deposit/withdrawal instructions.

        For manual mode, this just logs what needs to be done.
        The user must perform the actual deposit/withdrawal.

        Args:
            state: Current portfolio state
            trades: Trades to prepare
            routing_state: Routing state details
            **kwargs: Additional arguments
        """
        for trade in trades:
            freqtrade_id = trade.pair.other_data["freqtrade_id"]
            config = self.freqtrade_configs[freqtrade_id]

            if config.deposit_method == FreqtradeDepositMethod.manual:
                # Mark trade as awaiting manual confirmation
                amount = trade.planned_reserve if trade.planned_reserve else trade.planned_quantity
                trade.notes = (
                    f"Manual deposit required: Transfer {amount} {config.reserve_currency} "
                    f"to {config.exchange} for Freqtrade instance '{freqtrade_id}'"
                )
                logger.info(f"Trade {trade.trade_id}: {trade.notes}")
            else:
                raise NotImplementedError(
                    f"Deposit method {config.deposit_method} not yet implemented"
                )

    def settle_trade(
        self,
        state: State,
        trade: TradeExecution,
        **kwargs,
    ):
        """Settle a trade after execution.

        For manual mode, the user confirms when deposit/withdrawal is complete
        and this method would verify it via API.

        Args:
            state: Current portfolio state
            trade: Trade to settle
            **kwargs: Additional arguments
        """
        freqtrade_id = trade.pair.other_data["freqtrade_id"]
        config = self.freqtrade_configs[freqtrade_id]

        if config.deposit_method == FreqtradeDepositMethod.manual:
            # In manual mode, settling is manual too
            # In future, we could query Freqtrade API to verify balance changed
            logger.info(f"Settling manual trade {trade.trade_id} for Freqtrade {freqtrade_id}")
        else:
            raise NotImplementedError(
                f"Deposit method {config.deposit_method} not yet implemented"
            )
