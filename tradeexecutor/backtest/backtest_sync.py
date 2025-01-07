import logging
import datetime
from dataclasses import dataclass
from decimal import Decimal
from types import NoneType
from typing import List, Optional, Collection, Iterable

from web3.types import BlockIdentifier

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.types import JSONHexAddress, BlockNumber
from tradeexecutor.strategy.interest import prepare_interest_distribution, \
    accrue_interest
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.sync_model import SyncModel, OnChainBalance
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.ethereum.balance_update import apply_reserve_update_events
from tradingstrategy.chain import ChainId
from tradingstrategy.utils.time import ZERO_TIMEDELTA

logger = logging.getLogger(__name__)


@dataclass
class FundFlowEvent:
    """One simulated deposit/redemption in backtest.

    - Events can be triggered any time with :py:meth:`BacktestSyncModel.simulate_funding`

    - Fund flow is added to the reserves during :py:meth:`BacktestSyncModel.sync_treasury`
      as it would be with live trading
    """


    timestamp: datetime.datetime

    amount: Decimal


class BacktestSyncModel(SyncModel):
    """Backtest sync model.

    Simulate deposit events to the backtest wallet.

    - Read on-chain simulated wallet and reflect its balances back to the state
    """

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal | None = None):
        self.wallet = wallet

        #: Simulated deposit/redemption events pending to be processed
        #:
        #: Legacy code path.
        #:
        self.fund_flow_queue: List[FundFlowEvent] = []
        if initial_deposit_amount is not None:
            assert isinstance(initial_deposit_amount, Decimal)
            if initial_deposit_amount > 0:
                self.fund_flow_queue.append(FundFlowEvent(datetime.datetime.utcnow(), initial_deposit_amount))

    def get_token_storage_address(self) -> Optional[JSONHexAddress]:
        return None

    def sync_initial(self, state: State):
        """Set up the initial sync details.

        For backtesting these are irrelevant.
        """
        deployment = state.sync.deployment
        deployment.chain_id = None
        deployment.address = None
        deployment.block_number = None
        deployment.tx_hash = None
        deployment.block_mined_at = None
        deployment.vault_token_name = None
        deployment.vault_token_symbol = None

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: Optional[List[AssetIdentifier]] = None,
        end_block: BlockNumber | NoneType = None,
        post_valuation=False,
    ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle.

        .. warning::

            Old legacy code with wrong return signature compared to the parent class
        """

        assert len(supported_reserves) == 1
        reserve_token = supported_reserves[0]

        assert end_block is None, "Cannot use block ranges with backtesting"

        reserve_update_events = []  # TODO: Legacy

        for funding_event in self.fund_flow_queue:

            past_balance = self.wallet.get_balance(reserve_token.address)

            # Update wallet
            self.wallet.update_token_info(reserve_token)
            self.wallet.update_balance(reserve_token.address, funding_event.amount)

            # Generate a deposit event
            reserve_update_events.append(
                ReserveUpdateEvent(
                    asset=reserve_token,
                    past_balance=past_balance,
                    new_balance=self.wallet.get_balance(reserve_token.address),
                    updated_at=strategy_cycle_ts,
                    mined_at=funding_event.timestamp,
                )
            )

        balance_update_events = apply_reserve_update_events(state, reserve_update_events)

        # Clear our pending funding simulation events
        self.fund_flow_queue = []

        return balance_update_events

    def simulate_funding(self, timestamp: datetime.datetime, amount: Decimal):
        """Simulate a funding flow event.

        Call for the test to cause deposit or redemption for the backtest.
        The event goes to a queue and is processed in next `tick()`
        through `sync_portfolio()`.

        :param amount:
            Positive for deposit, negative for redemption

        """
        self.fund_flow_queue.append(
            FundFlowEvent(timestamp, amount)
        )

    def create_transaction_builder(self) -> None:
        """Backtesting does not need to care about how to build blockchain transactions."""

    def calculate_accrued_interest(
        self,
        strategy_universe: TradingStrategyUniverse,
        asset: AssetIdentifier,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> Decimal:
        """Calculate accrued interest of a position since last update."""

        lending_universe = strategy_universe.data_universe.lending_candles

        if asset.is_credit():
            candles = lending_universe.supply_apr
        elif asset.is_debt():
            candles = lending_universe.variable_borrow_apr
        else:
            raise AssertionError(f"Does not know how an asset behaves and lending markets {asset}")

        reserve = strategy_universe.data_universe.lending_reserves.get_by_chain_and_address(
            ChainId(asset.chain_id),
            asset.underlying.address
        )

        return candles.estimate_accrued_interest(reserve, start, end)

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> List[BalanceUpdate]:

        assert isinstance(timestamp, datetime.datetime)
        if not universe.has_lending_data():
            # sync_interests() is not needed for backtesting that do not deal with
            # leveraged positions
            return []

        previous_update_at = state.sync.interest.last_sync_at
        if not previous_update_at:
            # No interest based positions yet?
            logger.info(f"Interest sync checkpoint not set at {timestamp}, nothing to sync/cannot sync interest.")
            return []

        duration = timestamp - previous_update_at
        if duration == ZERO_TIMEDELTA:
            # TODO: Promote to warning and fix the cause
            logger.info(f"Sync time span must be positive:{previous_update_at} - {timestamp}")
            return []

        logger.info(
            "Starting backtest interest distribution operation at: %s, previous update %s, syncing %s",
            timestamp,
            previous_update_at,
            duration,
        )

        interest_distribution = prepare_interest_distribution(
            state.sync.interest.last_sync_at,
            timestamp,
            state.portfolio,
            pricing_model
        )

        # initialise_tracking(portfolio_interest_tracker, interest_distribution)

        # First simulate balances going up in the wallet
        for asset in interest_distribution.assets:
            accrued_multiplier = self.calculate_accrued_interest(
                universe,
                asset,
                previous_update_at,
                timestamp,
            )

            old_amount = self.wallet.get_balance(asset)
            self.wallet.rebase(asset, old_amount * accrued_multiplier)

        # Then sync interest "back from the chain"
        balances = {}
        for asset in interest_distribution.assets:
            balances[asset] = self.wallet.get_balance(asset)

        # Then distribute gained interest (new atokens/vtokens)
        # among positions
        events_iter = accrue_interest(state, balances, interest_distribution, timestamp, None)

        events = list(events_iter)

        return events

    def fetch_onchain_balances(
            self,
            assets: Collection[AssetIdentifier],
            filter_zero=True,
            block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:
        raise NotImplementedError("Backtesting does not know about on-chain balances")