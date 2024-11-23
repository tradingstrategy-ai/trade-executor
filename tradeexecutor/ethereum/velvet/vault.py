"""Velvet vault integration."""

import logging
import datetime
from typing import Optional

from eth_typing import HexAddress

from eth_defi.hotwallet import HotWallet
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.balance_update import apply_balance_update_events

from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.types import JSONHexAddress, USDollarPrice
from tradeexecutor.strategy.asset import build_expected_asset_map
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId

from tradingstrategy.pair import PandasPairUniverse

logger = logging.getLogger(__name__)


class VelvetVaultSyncModel(AddressSyncModel):
    """Update Velvet vault balances.

    - Velvet smart contract automatically converts any deposits into position balances.
      User can deposit any token, but those tokens will get traded into open positions.

    - Velvet does only in-kind redemptions
    """

    def __init__(
        self,
        pair_universe: PandasPairUniverse,
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
        self.pair_universe = pair_universe
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
        self,
        state: State,
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

    def sync_positions(
        self,
        timestamp: datetime.datetime,
        state: State,
        strategy_universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ):
        """Detect any position balance changes due to deposit/redemptions of vault users.

        - Velvet directly trades any incoming tokens to user balances

        - USDC/reserve token is synced by :py:meth:`sync_treasury`
        """

        # assets = get_relevant_assets(pair_universe, reserve_assets, state)
        asset_to_position_map = build_expected_asset_map(
            state.portfolio,
            pair_universe=strategy_universe.data_universe.pairs,
            ignore_reserve=True,
        )

        logger.info(
            "Velvet sync_positions(), %d assets to consider",
            len(asset_to_position_map),
        )

        # Some sample output
        for key, value in list(asset_to_position_map.items())[:5]:
            logger.info("Asset %s: %s", key, value)

        balances = self.fetch_onchain_balances(
            list(asset_to_position_map.keys())
        )

        # Apply any deposit/redemptions on positions
        events = apply_balance_update_events(
            timestamp,
            strategy_universe,
            state,
            pricing_model,
            balances,
            asset_to_position_map,
        )

        return events


