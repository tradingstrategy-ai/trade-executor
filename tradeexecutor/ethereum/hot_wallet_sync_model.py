"""Sync model for strategies using a single hot wallet."""
import datetime
from typing import List, Optional

from eth_defi.hotwallet import HotWallet
from tradeexecutor.state.balance_update import BalanceUpdate
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.identifier import AssetIdentifier

from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.testing.dummy_wallet import apply_sync_events


class HotWalletSyncModel(SyncModel):
    """V0 prototype version of sync model, only for.

    .. warning::

        TODO: This model is unfinished and needs to be migrated to event based.

    """

    def __init__(self, web3: Web3, hot_wallet: HotWallet):
        self.web3 = web3
        self.hot_wallet = hot_wallet

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def sync_initial(self, state: State):
        """Set u[ initial sync details."""

        web3 = self.web3
        deployment = state.sync.deployment
        deployment.chain_id = ChainId(web3.eth.chain_id)
        deployment.address = self.hot_wallet.address
        deployment.block_number = web3.eth.block_number
        deployment.tx_hash = None
        deployment.block_mined_at = datetime.datetime.utcnow()
        deployment.vault_token_name = None
        deployment.vault_token_symbol = None

    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 supported_reserves: Optional[List[AssetIdentifier]] = None
                 ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle."""

        # TODO: This code is not production ready - use with care
        # Needs legacy cleanup
        events = sync_reserves(self.web3, strategy_cycle_ts, self.hot_wallet.address, [], supported_reserves)
        apply_sync_events(state.portfolio, events)
        state.sync.treasury.last_updated_at = datetime.datetime.utcnow()
        state.sync.treasury.last_cycle_at = strategy_cycle_ts
        state.sync.treasury.last_block_scanned = self.web3.eth.block_number
        state.sync.treasury.balance_update_refs = []  # Broken - wrong event type
        return []

    def create_transaction_builder(self) -> HotWalletTransactionBuilder:
        return HotWalletTransactionBuilder(self.web3, self.hot_wallet)

    def setup_all(self, state: State, supported_reserves: List[AssetIdentifier]):
        """Make sure we have everything set up and initial test balance synced.
        
        A shortcut used in testing.
        """
        self.init()
        self.sync_initial(state)
        self.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves)


def EthereumHotWalletReserveSyncer(
     strategy_cycle_ts: datetime.datetime,
     state: State,
     supported_reserves: Optional[List[AssetIdentifier]] = None
):
    """Version 0 legacy.

    Do not use.
    """
    raise NotImplementedError()
    events = sync_reserves(self.web3, strategy_cycle_ts, self.tx_builder.address, [], supported_reserves)
    apply_sync_events(state.portfolio, events)
    return events

