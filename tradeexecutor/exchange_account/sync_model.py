"""Exchange account sync model for external perp DEXes.

Syncs account value changes from external exchanges (Derive, Hyperliquid, etc.)
and generates BalanceUpdate events for profit/loss tracking.
"""

import datetime
import logging
from decimal import Decimal
from typing import Callable, Collection, Iterable, List

from web3.types import BlockIdentifier

from eth_defi.hotwallet import HotWallet

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import JSONHexAddress, BlockNumber
from tradeexecutor.strategy.sync_model import SyncModel, OnChainBalance
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.pricing_model import PricingModel

logger = logging.getLogger(__name__)


class ExchangeAccountSyncModel(SyncModel):
    """Sync exchange account positions from external perp DEXes.

    Detects account value changes between syncs and generates BalanceUpdate
    events to track profit/loss from trading on the external exchange.

    The account value function is pluggable to support different exchanges
    (Derive, Hyperliquid, etc.).

    Example:

    .. code-block:: python

        from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
        from tradeexecutor.exchange_account.derive import create_derive_account_value_func

        # Create Derive-specific account value function
        clients = {subaccount_id: derive_client}
        account_value_func = create_derive_account_value_func(clients)

        # Create sync model
        sync_model = ExchangeAccountSyncModel(account_value_func)

        # Sync positions to detect PnL changes
        events = sync_model.sync_positions(timestamp, state, universe, pricing_model)
    """

    def __init__(
        self,
        account_value_func: Callable[[TradingPairIdentifier], Decimal],
    ):
        """Initialise sync model.

        :param account_value_func:
            Function that takes a TradingPairIdentifier and returns the current
            account value in USD from the exchange API.
        """
        self.account_value_func = account_value_func

    def has_position_sync(self) -> bool:
        """We sync positions, not treasury."""
        return True

    def get_token_storage_address(self) -> JSONHexAddress | None:
        """Tokens are stored on the exchange, not on-chain."""
        return None

    def get_hot_wallet(self) -> HotWallet | None:
        """No hot wallet needed for exchange account sync."""
        return None

    def sync_initial(self, state: State, **kwargs):
        """Initialise sync state.

        For exchange accounts, no special initialisation is needed.
        """
        pass

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: List[AssetIdentifier] | None = None,
        end_block: BlockNumber | None = None,
        post_valuation: bool = False,
    ) -> List[BalanceUpdate]:
        """No treasury sync needed for exchange accounts.

        Exchange account positions are synced via sync_positions().
        """
        return []

    def sync_positions(
        self,
        timestamp: datetime.datetime,
        state: State,
        strategy_universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> list[BalanceUpdate]:
        """Detect account value changes and generate balance updates.

        Compares the current account value from the exchange API with the
        tracked position quantity. If there's a difference (profit or loss),
        creates a BalanceUpdate event to record the change.

        :param timestamp:
            Current timestamp for the sync
        :param state:
            Strategy state containing positions
        :param strategy_universe:
            Trading universe (not used for exchange accounts)
        :param pricing_model:
            Pricing model (not used for exchange accounts)
        :return:
            List of BalanceUpdate events for positions with value changes
        """
        events = []

        for position in state.portfolio.get_open_positions():
            if not position.is_exchange_account():
                continue

            try:
                # Get current account value from exchange API
                current_value = self.account_value_func(position.pair)
            except Exception as e:
                logger.error(
                    "Failed to get account value for position %d: %s",
                    position.position_id,
                    e,
                )
                continue

            # Get tracked value (sum of trades + previous balance updates)
            tracked_value = position.get_quantity()

            # Calculate difference (PnL)
            diff = current_value - tracked_value

            if diff == 0:
                logger.debug(
                    "Exchange account position %d: no change (value=%.2f)",
                    position.position_id,
                    current_value,
                )
                continue

            logger.info(
                "Exchange account position %d: value changed %.2f -> %.2f (diff=%.2f)",
                position.position_id,
                tracked_value,
                current_value,
                diff,
            )

            # Allocate event ID
            event_id = state.portfolio.next_balance_update_id
            state.portfolio.next_balance_update_id += 1

            # Create balance update event
            evt = BalanceUpdate(
                balance_update_id=event_id,
                position_type=BalanceUpdatePositionType.open_position,
                cause=BalanceUpdateCause.vault_flow,
                asset=position.pair.base,
                block_mined_at=timestamp,
                strategy_cycle_included_at=timestamp,
                chain_id=position.pair.base.chain_id,
                old_balance=tracked_value,
                usd_value=float(diff),
                quantity=diff,
                owner_address=None,
                tx_hash=None,
                log_index=None,
                position_id=position.position_id,
                block_number=None,
                notes=f"Exchange account PnL sync: {position.pair.get_exchange_account_protocol()}",
            )

            # Store on position
            position.balance_updates[evt.balance_update_id] = evt

            # Track in accounting
            ref = BalanceEventRef.from_balance_update_event(evt)
            state.sync.accounting.balance_update_refs.append(ref)

            events.append(evt)

        if events:
            state.sync.accounting.last_updated_at = datetime.datetime.utcnow()

        return events

    def fetch_onchain_balances(
        self,
        assets: Collection[AssetIdentifier],
        filter_zero: bool = True,
        block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:
        """Not applicable for exchange accounts.

        Exchange account balances are fetched via the account_value_func,
        not via on-chain queries.
        """
        return []

    def create_transaction_builder(self) -> TransactionBuilder | None:
        """No transaction builder needed.

        Exchange account trades are executed on the exchange, not on-chain.
        """
        return None
