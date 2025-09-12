"""Multistage deposit redeem flows for trading positions.

- Vault positions like Lagoon and Gains require multiple transactions to deposit and redeem

"""
import datetime
from functools import cached_property

from dataclasses_json import dataclass_json, config
from dataclasses import dataclass, field

from web3 import Web3

from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.vault.base import VaultBase
from eth_defi.vault.deposit_redeem import DepositTicket, RedemptionTicket, VaultDepositManager, DepositRequest, RedemptionRequest
from tradeexecutor.ethereum.vault.vault_routing import get_vault_for_pair

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json, decode_pickle_over_json
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, MultiStageTrade
from tradeexecutor.state.types import JSONHexAddress, BlockNumber
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradingstrategy.types import USDollarAmount
from marshmallow import fields


@dataclass_json
@dataclass(slots=True)
class TicketState:
    """Capture data of multi transaction ongoing deposit/redeem process.

    Stored as `TradeExecution.other_data["multi_stage_state"]`
    """

    #: If this is a multi-stage trade, what was the first trade that started the operation.
    #:
    #: Set to None on the first trade.
    first_part_trade_id: int | None = None

    #: Not serialised, only passed until the tx is complete
    deposit_request: DepositRequest = fields.Raw(
        dump_only=False,  # Allow decoding
        load_only=True,   # Exclude from encoding
        load_default=None,  # Always decode to None
        default=None,
    )

    deposit_ticket: DepositTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        ),
        default=None,
    )

    #: Not serialised, only passed until the tx is complete
    redemption_request: RedemptionRequest = fields.Raw(
        dump_only=False,  # Allow decoding
        load_only=True,   # Exclude from encoding
        load_default=None,  # Always decode to None
        default=None,
    )
    redemption_ticket: RedemptionTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        ),
        default=None,
    )

    def is_in_progress(self):
        return self.first_part_trade_id is None


@dataclass_json
@dataclass(slots=True)
class DepositTicketState(TicketState):
    """Encapsulate raw eth_defi data structures"""

    ticket: DepositTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        ),
        default=None,
    )

    def can_finish(self, deposit_manager: VaultDepositManager) -> bool:
        """Can we finish this deposit now?"""
        return deposit_manager.can_finish_deposit(self.ticket)



@dataclass_json
@dataclass(slots=True)
class RedemptionTicketState(TicketState):
    """Encapsulate raw eth_defi data structures"""
    ticket: RedemptionTicket = field(
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        ),
        default=None,
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

    def can_start(self, deposit_manager: VaultDepositManager) -> bool:
        """Gains vault allows us to start redemption on certain days only"""
        return deposit_manager.can_start_deposit()


@dataclass_json
@dataclass(slots=True)
class DepositTicketQueue:
    """Manage state of our various multi-stage deposits and redeems."""
    queue: list[DepositTicketState] = field(default_factory=list)



@dataclass_json
@dataclass(slots=True)
class MultiStageState:
    """Manage state of our various multi-stage deposits and redeems."""
    vault_feature_flags: set[ERC4626Feature] = field(default_factory=set)
    deposits: TicketQueue = field(default_factory=DepositTicketQueue)
    redeems: TicketQueue = field(default_factory=RedemptionTicketQueue)


@dataclass_json
@dataclass(slots=True)
class MultiStageTradeState:
    """Manage state of our various multi-stage deposits and redeems."""


class MultiStageDepositRedeemManager:
    """Helper for multistage deposit and redeem flows.

    - For a single vault
    - Manage ins and outs with deposit() adn claim() cycles of ERC-7540
    """

    def __init__(
        self,
        web3: Web3,
        pair: TradingPairIdentifier,
        trading_address: JSONHexAddress,
        position_manager: PositionManager,
    ):
        """
        :param web3:
            Web3 instance
        :param trading_address:
            Our address we use for trading: vault or hot wallet
        """

        assert isinstance(pair, TradingPairIdentifier)
        assert pair.is_vault()

        self.pair = pair
        self.web3 = web3

        # Store to speed up the future institiation
        self.vault_feature_flags = pair.get_vault_features()

        self.trading_address = trading_address
        self.position_manager = position_manager

    @cached_property
    def vault_contract_address(self) -> JSONHexAddress:
        return self.pair.pool_address

    @cached_property
    def vault(self) -> VaultBase:
        return create_vault_instance(
            self.web3,
            self.vault_contract_address,
            features=self.state.vault_feature_flags,
        )

    @cached_property
    def deposit_manager(self) -> VaultDepositManager:
        return self.vault.get_deposit_manager()

    def can_start_deposit(self) -> bool:
        return self.deposit_manager.can_create_deposit_request(self.trading_address)


    def budge(self) -> TradeExecution:
        """Create blockchain transactions for all pending deposits/redeems that can be finished now.

        - We will create zero-valued trades to finish the deposits/redeems
        """
        pass


class MultiStagePositionWrapper:
    """Helper class to manager state for a position"""

    def __init__(self, manager: MultiStageDepositRedeemManager, position: "tradeexecutor.state.position.TradingPosition"):
        from tradeexecutor.state.position import TradingPosition
        assert isinstance(manager, MultiStageDepositRedeemManager)
        assert isinstance(position, TradingPosition)
        self.manager = manager
        self.position = position

    def get_state(self) -> MultiStageState | None:
        return self.position.other_data.get("multi_stage_deposit_redeem", MultiStageState())

    @cached_property
    def state(self) -> MultiStageState:
        return self.get_state()

    def is_deposit_in_progress(self) -> bool:
        return self.state.deposits.is_in_progress()

    def is_redemption_in_progress(self) -> bool:
        return self.state.redeems.is_in_progress()

    def mark_deposit_in_progress(
        self,
        deposit_ticket: DepositTicket,
    ):
        last_ticket = self.state.deposits.queue[-1]
        assert not last_ticket.is_in_progress(), f"No open deposit in progress: {last_ticket}"
        ticket = DepositTicketState(ticket=deposit_ticket)
        ticket.opened_at = datetime.datetime.utcnow()
        ticket.opened_block = ticket.block_number
        ticket.opened_tx_hash = ticket.tx_hash
        self.state.deposits.append(ticket)

    def mark_deposit_finished(
        self,
        tx_hash: str,
        block_number: int,
        block_time: datetime.datetime,
    ):
        last_ticket = self.state.deposits.queue[-1]
        assert last_ticket.is_in_progress(), f"No open deposit in progress: {last_ticket}"
        self.state.closed_at = block_time
        self.state.closed_block = block_number
        self.state.closed_tx_hash = tx_hash

    def can_start_redeem(self, deposit_manager: VaultDepositManager) -> bool:
        if self.is_deposit_in_progress():
            # We do not suppor cancelling of deposit requests yet
            return False
        return self.deposit_manager.create_redemption_request(self.trading_address)


def get_or_initialise_multi_stage_state(trade: TradeExecution) -> TicketState:
    if "multi_stage_state" not in trade.other_data:
        trade.other_data["multi_stage_state"] = TicketState()
    return trade.other_data["multi_stage_state"]


def get_multi_stage_state(trade: TradeExecution) -> TicketState:
    assert "multi_stage_state" in trade.other_data, f"Trade {trade} did not have multi stage structure initialised"
    state = trade.other_data["multi_stage_state"]
    return state


def mark_position_multi_stage(
    position: "tradeexecutor.state.position.TradingPosition",
):
    position.other_data["multi_stage"] = True


def mark_trade_multi_stage_deposit_requested(
    trade: TradeExecution,
    deposit_request: DepositRequest,
):
    state = get_or_initialise_multi_stage_state(trade)
    state.deposit_request = deposit_request


def mark_trade_multi_stage_deposit_started(
    trade: TradeExecution,
    deposit_ticket: DepositTicket,
):
    assert isinstance(deposit_ticket, DepositTicket)
    state = get_multi_stage_state(trade)
    state.deposit_ticket = deposit_ticket


def start_multi_stage_deposit(
    position_manager: PositionManager,
    pair: TradingPairIdentifier,
    amount: USDollarAmount,
    notes: str | None = None,
) -> TradeExecution:
    """Start a deposit process.

    - Creates an open position for which we populate the multi-stage deposit structure

    - Buy trades are deposit, sell trades are redemptions
    """
    assert amount > 0

    position = position_manager.get_current_position_for_pair(pair, pending=False)
    assert not position, f"Already have an open position for {pair}: {position}"

    pair = pair
    quantity_delta = None

    trades = position_manager.adjust_position(
        pair=pair,
        dollar_delta=amount,
        quantity_delta=quantity_delta,
        notes=notes,
        weight=1,
    )

    assert len(trades) == 1
    trade = trades[0]

    ticket_state = get_or_initialise_multi_stage_state(trade)
    assert ticket_state is not None

    position = position_manager.state.portfolio.open_positions[trade.position_id]
    mark_position_multi_stage(position)

    return trade


def can_complete_multi_stage(
    web3: Web3,
    position: TradingPosition,
):
    """Check if the pre-conditions are filled that we can perform the second leg transaction of multi-stage deposit/redeem.

    - Checks onchain if Lagoon vault has settled, Gains timeout passed, etc.
    """

    assert position.is_multi_stage_in_process(), f"Position does not have multi-stage in progress: {position}"

    vault = get_vault_for_pair(
        web3,
        position.pair,
    )

    state = get_multi_stage_state(position.get_last_trade())

    deposit_manager = vault.deposit_manager

    match position.get_multi_stage_phase():
        case "deposit":
            ticket = state.deposit_ticket
            assert ticket, f"No deposit_ticket: {state}"
            return deposit_manager.can_finish_deposit(state)
        case "redeem":
            ticket = state.redemption_ticket
            assert ticket, f"No redemption_ticket: {state}"
            return deposit_manager.can_finish_redeem(ticket)
