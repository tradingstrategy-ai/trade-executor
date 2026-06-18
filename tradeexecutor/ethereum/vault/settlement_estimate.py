"""Display-only async vault settlement estimate helpers."""

import logging

from tradeexecutor.state.trade import TradeExecution


logger = logging.getLogger(__name__)


def refresh_vault_settlement_estimate(
    trade: TradeExecution,
    deposit_manager,
    ticket,
    direction: str,
) -> None:
    """Refresh display-only vault settlement ETA from a protocol ticket.

    Some async vault adapters can estimate when a specific request ticket
    becomes eligible for settlement. Persist that estimate in ``other_data`` so
    all user interfaces show the same metadata.
    """
    method_name = (
        "get_deposit_ticket_delay_over"
        if direction == "deposit"
        else "get_redemption_ticket_delay_over"
    )
    method = getattr(deposit_manager, method_name, None)
    if method is None:
        return

    try:
        settles_at = method(ticket)
    except NotImplementedError:
        return
    except Exception as e:
        logger.warning(
            "Could not refresh vault settlement estimate for trade #%d: %s",
            trade.trade_id,
            e,
        )
        return

    trade.other_data["vault_settlement_estimated_at"] = (
        settles_at.isoformat() if settles_at else None
    )
