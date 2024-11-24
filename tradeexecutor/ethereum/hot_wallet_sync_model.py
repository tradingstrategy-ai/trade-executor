"""Sync model for strategies using a single hot wallet."""
import datetime
import logging
from decimal import Decimal
from types import NoneType
from typing import List, Optional, Iterable

from web3.types import BlockIdentifier

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.types import JSONHexAddress
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.ethereum.balance_update import apply_reserve_update_events



logger = logging.getLogger(__name__)


class HotWalletSyncModel(AddressSyncModel):
    """V0 prototype version of sync model, only for.

    .. warning::

        TODO: This model is unfinished and needs to be migrated to event based.

    """

    def __init__(self, web3: Web3, hot_wallet: HotWallet):
        self.web3 = web3
        self.hot_wallet = hot_wallet

    def init(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_hot_wallet(self) -> Optional[HotWallet]:
        return self.hot_wallet

    def get_token_storage_address(self) -> Optional[str]:
        return self.hot_wallet.address

    def get_main_address(self) -> Optional[JSONHexAddress]:
        return self.hot_wallet.address

    def resync_nonce(self):
        self.hot_wallet.sync_nonce(self.web3)

    def create_transaction_builder(self) -> HotWalletTransactionBuilder:
        return HotWalletTransactionBuilder(self.web3, self.hot_wallet)


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
    apply_reserve_update_events(state.portfolio, events)
    return events

