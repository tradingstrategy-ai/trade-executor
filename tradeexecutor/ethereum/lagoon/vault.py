"""Velvet vault integration."""

import logging
import datetime
from typing import Optional

from eth_typing import HexAddress

from eth_defi.hotwallet import HotWallet
from eth_defi.lagoon.vault import LagoonVault
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.balance_update import apply_balance_update_events
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder

from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.balance_update import BalanceUpdate
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import JSONHexAddress, USDollarPrice
from tradeexecutor.strategy.asset import build_expected_asset_map
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId


logger = logging.getLogger(__name__)


class LagoonVaultSyncModel(AddressSyncModel):
    """Update Lagoon vault balances.

    - We do specific NAV update and settlement cycle to update
    """

    def __init__(
        self,
        vault: LagoonVault,
        hot_wallet: HotWallet | None,
        extra_gnosis_gas: int,
    ):
        self.vault = vault
        self.hot_wallet = hot_wallet
        self.extra_gnosis_gas = extra_gnosis_gas

    def __repr__(self):
        return f"<LagoonVaultSyncModel for vault {self.vault.name} ({self.vault_address})>"

    @property
    def web3(self):
        return self.vault.web3

    @property
    def portfolio_address(self) -> HexAddress:
        return self.vault.spec.vault_address

    @property
    def vault_address(self) -> HexAddress:
        return self.vault.address

    @property
    def chain_id(self) -> ChainId:
        return ChainId(self.vault.spec.chain_id)

    def get_hot_wallet(self) -> Optional[HotWallet]:
        return self.hot_wallet

    def get_key_address(self) -> Optional[str]:
        return self.vault.vault_address

    def get_main_address(self) -> Optional[JSONHexAddress]:
        return self.vault.vault_address

    def get_token_storage_address(self) -> Optional[str]:
        return self.vault.safe_address

    def create_transaction_builder(self) -> LagoonTransactionBuilder:
        return LagoonTransactionBuilder(self.vault, self.hot_wallet, self.extra_gnosis_gas)

    def sync_initial(
        self,
        state: State,
        reserve_asset: AssetIdentifier | None = None,
        reserve_token_price: USDollarPrice | None = None,
        **kwargs,
    ):
        """Set ups sync starting point"""
        super().sync_initial(
            state=state,
            reserve_asset=reserve_asset,
            reserve_token_price=reserve_token_price,
        )

        deployment = state.sync.deployment
        deployment.vault_token_name = self.vault.name
        deployment.vault_token_symbol = self.vault.symbol
