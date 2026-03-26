"""Sync model for strategies which poll blockchain for incoming deposits/withdrawals."""
import datetime
import logging
from abc import abstractmethod
from types import NoneType
from typing import List, Optional, Iterable

from web3.types import BlockIdentifier

from eth_defi.compat import native_datetime_utc_now
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from tradeexecutor.ethereum.multichain_balance import fetch_onchain_balances_multichain
from tradeexecutor.state.balance_update import BalanceUpdate
from tradingstrategy.chain import ChainId
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import BlockNumber, JSONHexAddress, USDollarPrice
from tradeexecutor.strategy.sync_model import SyncModel, OnChainBalance
from tradeexecutor.strategy.interest import (
    sync_interests,
)
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.ethereum.balance_update import apply_reserve_update_events

logger = logging.getLogger(__name__)


class AddressSyncModel(SyncModel):
    """Sync vault/wallet address by polling its list of assets.

    Supports optional multichain balance queries via :py:attr:`web3config`.
    """

    #: Optional Web3Config with connections to multiple chains.
    #: When set, :py:meth:`fetch_onchain_balances` can query satellite chain balances.
    web3config: "Web3Config | None" = None

    @abstractmethod
    def get_token_storage_address(self) -> Optional[JSONHexAddress]:
        """Which is the onchain address having our token balances."""

    @abstractmethod
    def get_main_address(self) -> Optional[JSONHexAddress]:
        """Which is the onchain address that identifies this wallet/vault deployment.

        See also :py:meth:`get_token_storage_address`
        """

    def sync_initial(
        self, state: State,
        reserve_asset: AssetIdentifier | None = None,
        reserve_token_price: USDollarPrice | None = None,
        **kwargs,
    ):
        """Set up the initial sync details.

        - Initialise vault deployment information

        - Set the reserve assets (optional, sometimes can be read from the chain)

        """

        assert isinstance(reserve_asset, (AssetIdentifier, NoneType)), f"Got {type(reserve_asset)}"

        web3 = self.web3
        deployment = state.sync.deployment
        deployment.chain_id = ChainId(web3.eth.chain_id)
        deployment.address = self.get_main_address()
        deployment.block_number = web3.eth.block_number
        deployment.tx_hash = None
        deployment.block_mined_at = native_datetime_utc_now()
        deployment.vault_token_name = None
        deployment.vault_token_symbol = None
        deployment.initialised_at = native_datetime_utc_now()

        if reserve_asset:
            position = state.portfolio.initialise_reserves(reserve_asset)
            position.last_pricing_at = native_datetime_utc_now()
            position.reserve_token_price = reserve_token_price

        logger.info(
            "Address sync model initialised, reserve asset is is %s, price is %s",
            reserve_asset,
            reserve_token_price,
        )

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: Optional[List[AssetIdentifier]] = None,
        end_block: BlockNumber | NoneType = None,
        post_valuation=False,
    ) -> List[BalanceUpdate]:
        """Poll chain for updated treasury token balances.

        - Apply the balance sync before each strategy cycle.

        TODO: end_block is being ignored
        """

        if supported_reserves is None:
            supported_reserves = [p.asset for p in state.portfolio.reserves.values()]

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

        # Map ReserveUpdateEvent (internal transitory) to BalanceUpdate events (persistent)
        balance_update_events = apply_reserve_update_events(state, events)

        treasury = state.sync.treasury
        treasury.last_updated_at = native_datetime_utc_now()
        treasury.last_cycle_at = strategy_cycle_ts
        treasury.last_block_scanned = block_number

        logger.info(f"Chain polling sync done, the last block is now {treasury.last_block_scanned:,}. got {len(events)} events")

        return balance_update_events

    def setup_all(self, state: State, supported_reserves: List[AssetIdentifier]):
        """Make sure we have everything set up and initial test balance synced.
        
        A shortcut used in testing.
        """
        self.init()
        self.sync_initial(state)
        self.sync_treasury(native_datetime_utc_now(), state, supported_reserves)

    def fetch_onchain_balances(
        self,
        assets: list[AssetIdentifier],
        filter_zero=True,
        block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:
        """Fetch on-chain token balances, with multichain support.

        When :py:attr:`web3config` is set, assets are grouped by ``chain_id``
        and each group is fetched from the correct chain's Web3 connection.
        Otherwise falls back to fetching all from ``self.web3``.

        Off-chain assets (e.g. Hypercore vault tokens on chain 9999) are
        silently filtered out via :py:func:`fetch_onchain_balances_multichain`.
        """

        address = self.get_token_storage_address()

        if self.web3config is not None and len(self.web3config.connections) > 1:
            # Multichain mode: route each asset group to the correct chain
            by_chain: dict[int, list[AssetIdentifier]] = {}
            for asset in assets:
                by_chain.setdefault(asset.chain_id, []).append(asset)

            for chain_id, chain_assets in by_chain.items():
                try:
                    chain_web3 = self.web3config.get_connection(ChainId(chain_id))
                except KeyError:
                    chain_web3 = None

                if chain_web3 is None:
                    # Fall back to default connection
                    chain_web3 = self.web3

                chain_block = get_almost_latest_block_number(chain_web3)
                logger.info(
                    "Fetching chain %s balances at block %d for %d asset(s)",
                    ChainId(chain_id).name,
                    chain_block,
                    len(chain_assets),
                )
                yield from fetch_onchain_balances_multichain(
                    chain_web3,
                    address,
                    chain_assets,
                    filter_zero=filter_zero,
                    block_number=chain_block,
                )
        else:
            # Single-chain mode (existing behaviour)
            if block_identifier is None:
                block_identifier = get_almost_latest_block_number(self.web3)

            yield from fetch_onchain_balances_multichain(
                self.web3,
                address,
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

