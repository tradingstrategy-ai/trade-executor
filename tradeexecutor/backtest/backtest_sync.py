import logging
import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from eth_defi.aave_v3.rates import SECONDS_PER_YEAR

from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.wallet import ReserveUpdateEvent
from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.interest import update_credit_supply_interest
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.dummy_wallet import apply_sync_events


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

    Simulate deposit events to the backtest wallet."""

    def __init__(self, wallet: SimulatedWallet, initial_deposit_amount: Decimal):
        assert isinstance(initial_deposit_amount, Decimal)
        self.wallet = wallet

        #: Simulated deposit/redemption events pending to be processed
        self.fund_flow_queue: List[FundFlowEvent] = []
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

    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle.

        .. warning::

            Old legacy code with wrong return signature compared to the parent class
        """

        assert len(supported_reserves) == 1
        reserve_token = supported_reserves[0]

        reserve_update_events = []  # TODO: Legacy

        for funding_event in self.fund_flow_queue:

            past_balance = self.wallet.get_balance(reserve_token.address)

            # Update wallet
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

        balance_update_events = apply_sync_events(state, reserve_update_events)

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
        universe: TradingStrategyUniverse,
        position: TradingPosition,
        timestamp: datetime.datetime,
    ) -> Decimal:
        """Calculate accrued interest of a position since last update."""
        # get relevant candles for the position period since last update until now
        collateral_interest = position.loan.collateral_interest
        previous_update_at = collateral_interest.last_event_at

        df = universe.universe.lending_candles.supply_apr.df.copy()
        supply_df = df[
            (df["timestamp"] >= previous_update_at)
            & (df["timestamp"] <= timestamp)
        ]

        if len(supply_df) == 0:
            # TODO: this is a temporary hack, we should make it better
            supply_df = df[
                (df["timestamp"] >= position.opened_at)
                & (df["timestamp"] <= timestamp)
            ]

        assert len(supply_df) > 0, f"No lending data for {position} from {previous_update_at} to {timestamp}"

        # get average APR from high and low
        supply_df["avg"] = supply_df[["high", "low"]].mean(axis=1)
        avg_apr = Decimal(supply_df["avg"].mean() / 100)

        amount = Decimal(collateral_interest.last_atoken_amount)
        duration = Decimal((timestamp - previous_update_at).total_seconds())
        accrued_interest_estimation = amount * avg_apr * duration / SECONDS_PER_YEAR

        return accrued_interest_estimation

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        positions: List[TradingPosition],
    ) -> List[BalanceUpdate]:

        assert universe.has_lending_data(), "Cannot update credit positions if no data is available"

        events = []
        for p in positions:

            if p.is_credit_supply():
                accrued = self.calculate_accrued_interest(
                    universe,
                    p,
                    timestamp,
                )

                interest = p.loan.collateral_interest

                # TODO: replace with a real interest calculation,
                # based on universe.lending_candles
                assert len(p.trades) <= 2, "This interest calculation does not support increase/reduce position"
                new_amount = interest.last_atoken_amount + accrued

                evt = update_credit_supply_interest(
                    state,
                    p,
                    p.pair.base,
                    new_atoken_amount=new_amount,
                    event_at=timestamp,
                )
                events.append(evt)

                # Make atokens magically appear in the simulated
                # backtest wallet. The amount must be updated, or
                # otherwise we get errors when closing the position.
                self.wallet.update_balance(p.pair.base.address, new_amount)

        return events
