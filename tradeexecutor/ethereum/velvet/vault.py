"""Enzyme vaults integration."""

import logging
import datetime
import pprint
from _decimal import Decimal
from functools import partial
from types import NoneType
from typing import cast, List, Optional, Tuple, Iterable

from eth_typing import HexAddress

from eth_defi.event_reader.conversion import convert_jsonrpc_value_to_int
from web3.types import BlockIdentifier

from eth_defi.provider.anvil import is_anvil
from eth_defi.provider.broken_provider import get_block_tip_latency, get_almost_latest_block_number
from web3 import Web3, HTTPProvider

from eth_defi.chain import fetch_block_timestamp, has_graphql_support
from eth_defi.enzyme.events import fetch_vault_balance_events, EnzymeBalanceEvent, Deposit, Redemption, fetch_vault_balances
from eth_defi.enzyme.vault import Vault
from eth_defi.event_reader.lazy_timestamp_reader import extract_timestamps_json_rpc_lazy, LazyTimestampContainer
from eth_defi.event_reader.reader import read_events, Web3EventReader, extract_events, extract_timestamps_json_rpc
from eth_defi.event_reader.reorganisation_monitor import ReorganisationMonitor
from eth_defi.hotwallet import HotWallet
from eth_defi.vault.base import VaultSpec
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import BlockNumber, JSONHexAddress, USDollarPrice
from tradeexecutor.strategy.dust import get_dust_epsilon_for_asset
from tradeexecutor.strategy.sync_model import SyncModel, OnChainBalance
from tradingstrategy.chain import ChainId
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.interest import sync_interests
from tradeexecutor.strategy.lending_protocol_leverage import reset_credit_supply_loan, update_credit_supply_loan

logger = logging.getLogger(__name__)


class UnknownAsset(Exception):
    """Cannot map redemption asset to any known position"""


class RedemptionFailure(Exception):
    """Cannot map redemption asset to any known position"""


class VelvetVaultSyncModel(AddressSyncModel):
    """Update Velvet vault balances."""

    def __init__(
        self,
        vault: VelvetVault,
        hot_wallet: HotWallet | None,
    ):
        """Connect to a Velvet vault for syncing the strategy positions and reserves.

        :param vault:
            Velvet low level Python interface

        :param hot_wallet:
            Asset manager private key who can perform trades.

            Leave empty if you only perform reads.

        :param reserve_asset:
            The vault reserve asset
        """
        self.vault = vault
        self.hot_wallet = hot_wallet

    def __repr__(self):
        return f"<VelvetVaultSyncModel for vault {self.vault.name} ({self.vault_address})>"

    @property
    def web3(self):
        return self.vault.web3

    @property
    def portfolio_address(self) -> HexAddress:
        return self.vault.spec.vault_address

    @property
    def vault_address(self) -> HexAddress:
        return self.vault.info["vaultAddress"]

    def get_main_address(self) -> Optional[JSONHexAddress]:
        """Which is the onchain address that identifies this wallet/vault deployment.

        See also :py:meth:`get_token_storage_address`
        """
        return self.vault_address

    @property
    def chain_id(self) -> ChainId:
        return ChainId(self.vault.spec.chain_id)

    def get_token_storage_address(self) -> Optional[str]:
        return self.vault_address

    def create_transaction_builder(self) -> VelvetTransactionBuilder:
        return VelvetTransactionBuilder(self.vault, self.hot_wallet)

    def sync_initial(
        self, state: State,
        reserve_asset: AssetIdentifier | None = None,
        reserve_token_price: USDollarPrice | None = None,
        **kwargs,
    ):
        super().sync_initial(
            state=state,
            reserve_asset=reserve_asset,
            reserve_token_price=reserve_token_price,
        )

        deployment = state.sync.deployment
        deployment.vault_token_name = self.vault.name
        deployment.vault_token_symbol = self.vault.token_symbol
