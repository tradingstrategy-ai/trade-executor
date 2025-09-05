"""Multistage deposit redeem flows for trading positions.

- Vault positions like Lagoon and Gains require multiple transactions to deposit and redeem

"""
import datetime
from functools import cached_property

from dataclasses_json import dataclass_json, config
from dataclasses import dataclass, field

from eth_defi.vault.deposit_redeem import DepositTicket, RedemptionTicket
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json, decode_pickle_over_json
from tradeexecutor.state.position import TradingPosition


@dataclass_json
@dataclass(slots=True)
class TicketState:
    opened_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    closed_at: datetime.datetime | None = None

    def is_in_progress(self) -> bool:
        return self.closed_at is None


@dataclass_json
@dataclass(slots=True)
class DepositTicketState(TicketState):
    """Encapsulate raw eth_defi data structures"""

    ticket: DepositTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        )
    )


@dataclass_json
@dataclass(slots=True)
class RedemptionTicketState(TicketState):
    """Encapsulate raw eth_defi data structures"""
    ticket: RedemptionTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        )
    )


@dataclass_json
@dataclass(slots=True)
class TicketQueue:
    """Manage state of our various multi-stage deposits and redeems."""
    queue: list[TicketState] = field(default_factory=list)

    def is_in_progress(self) -> bool:
        """Do we have an open multi-stage ticket in progress?"""
        ticket = self.queue[-1] if len(self.queue) > 0 else None
        if ticket is None:
            return False
        return ticket.is_in_progress()


@dataclass_json
@dataclass(slots=True)
class RedemptionTicketQueue:
    """Manage state of our various multi-stage deposits and redeems."""
    queue: list[RedemptionTicketState] = field(default_factory=list)



@dataclass_json
@dataclass(slots=True)
class DepositTicketQueue:
    """Manage state of our various multi-stage deposits and redeems."""
    queue: list[DepositTicketState] = field(default_factory=list)


@dataclass_json
@dataclass(slots=True)
class MultiStageState:
    """Manage state of our various multi-stage deposits and redeems."""
    deposits: TicketQueue = field(default_factory=DepositTicketQueue)
    redeems: TicketQueue = field(default_factory=RedemptionTicketQueue)


class MultiStageDepositRedeem:
    """Helper for multistage deposit and redeem flows."""

    def __init__(
        self,
        position: TradingPosition,
        router: GenericRouter,
    ):
        self.position = position
        self.router = router

    def get_state(self) -> MultiStageState | None:
        return self.position.other_data.get("multi_stage_deposit_redeem", MultiStageState())

    @cached_property
    def state(self) -> MultiStageState:
        return self.get_state()

    def is_deposit_in_progress(self) -> bool:
        return self.state.deposits.is_in_progress()

    def is_redemption_in_progress(self) -> bool:
        return self.state.redeems.is_in_progress()