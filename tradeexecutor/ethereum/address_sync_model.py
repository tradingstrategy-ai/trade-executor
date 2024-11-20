"""Sync model for strategies which poll blockchain for incoming deposits/withdrawals."""
import datetime
import logging
from abc import abstractmethod
from types import NoneType
from typing import List, Optional, Iterable

from web3.types import BlockIdentifier

from eth_defi.provider.broken_provider import get_almost_latest_block_number
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.state.balance_update import BalanceUpdate
from tradingstrategy.chain import ChainId
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import BlockNumber, JSONHexAddress
from tradeexecutor.strategy.sync_model import SyncModel, OnChainBalance
from tradeexecutor.strategy.interest import (
    sync_interests,
)
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.testing.dummy_wallet import apply_sync_events


logger = logging.getLogger(__name__)


class AddressSyncModel(SyncModel):
    """Sync vault/wallet address by polling its list of assests"""

    @abstractmethod
    def get_token_storage_address(self) -> Optional[JSONHexAddress]:
        """Which is the onchain address having our token balances."""

    @abstractmethod
    def get_main_address(self) -> Optional[JSONHexAddress]:
        """Which is the onchain address that identifies this wallet/vault deployment.

        See also :py:meth:`get_token_storage_address`
        """

    def sync_initial(self, state: State, **kwargs):
        """Set u[ initial sync details."""
        web3 = self.web3
        deployment = state.sync.deployment
        deployment.chain_id = ChainId(web3.eth.chain_id)
        deployment.address = self.get_main_address()
        deployment.block_number = web3.eth.block_number
        deployment.tx_hash = None
        deployment.block_mined_at = datetime.datetime.utcnow()
        deployment.vault_token_name = None
        deployment.vault_token_symbol = None
        deployment.initialised_at = datetime.datetime.utcnow()

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: Optional[List[AssetIdentifier]] = None,
        end_block: BlockNumber | NoneType = None,
    ) -> List[BalanceUpdate]:
        """Apply the balance sync before each strategy cycle.

        TODO: end_block is being ignored
        """

        # TODO: This code is not production ready - use with care
        # Needs legacy cleanup
        address = self.get_token_storage_address()
        logger.info("AddressSyncModel treasury sync starting for token hold address %s", address)
        assert address, f"Token storage address is None on {self}"
        current_reserves = list(state.portfolio.reserves.values())
        block_number = get_almost_latest_block_number(self.web3)
        events = sync_reserves(
            self.web3,
            strategy_cycle_ts,
            address,
            current_reserves,
            supported_reserves,
            block_identifier=block_number,
        )
        apply_sync_events(state, events)
        treasury = state.sync.treasury
        treasury.last_updated_at = datetime.datetime.utcnow()
        treasury.last_cycle_at = strategy_cycle_ts
        treasury.last_block_scanned = block_number
        treasury.balance_update_refs = []  # Broken - wrong event type
        logger.info(f"Hot wallet sync done, the last block is now {treasury.last_block_scanned:,}")
        return []

    def setup_all(self, state: State, supported_reserves: List[AssetIdentifier]):
        """Make sure we have everything set up and initial test balance synced.
        
        A shortcut used in testing.
        """
        self.init()
        self.sync_initial(state)
        self.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves)

    def fetch_onchain_balances(
        self,
        assets: List[AssetIdentifier],
        filter_zero=True,
        block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:

        # Latest block fails on LlamaNodes.com
        if block_identifier is None:
            block_identifier = get_almost_latest_block_number(self.web3)

        return fetch_address_balances(
            self.web3,
            self.get_hot_wallet().address,
            assets,
            filter_zero=filter_zero,
            block_number=block_identifier,
        )

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> List[BalanceUpdate]:

        return sync_interests(
            web3=self.web3,
            wallet_address=self.hot_wallet.address,
            timestamp=timestamp,
            state=state,
            universe=universe,
            pricing_model=pricing_model,
        )

