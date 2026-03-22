"""Velvet vault integration."""

import datetime
import logging
from decimal import Decimal
from pprint import pformat
from types import NoneType
from typing import Callable, Iterable, Optional

from web3 import Web3
from web3.contract.contract import ContractFunction

from eth_defi.compat import native_datetime_utc_now
from eth_defi.confirmation import wait_and_broadcast_multiple_nodes_mev_blocker
from eth_defi.erc_4626.vault_protocol.lagoon.analysis import \
    analyse_vault_flow_in_settlement
from eth_defi.erc_4626.vault_protocol.lagoon.vault import (
    DEFAULT_LAGOON_POST_VALUATION_GAS, DEFAULT_LAGOON_SETTLE_GAS, LagoonVault)
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import is_anvil
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from eth_defi.provider.mev_blocker import MEVBlockerProvider
from eth_defi.token import fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_typing import BlockIdentifier, HexAddress
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.state.balance_update import (BalanceUpdate,
                                                BalanceUpdateCause,
                                                BalanceUpdatePositionType)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import (BlockNumber, JSONHexAddress, Percent,
                                       USDollarAmount, USDollarPrice)
from tradeexecutor.strategy.interest import sync_interests
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.sync_model import OnChainBalance
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverse

logger = logging.getLogger(__name__)


def _transact_anvil_sequentially(
    web3,
    hot_wallet: HotWallet,
    txs: list[tuple[ContractFunction, int]],
    *,
    timeout: int = 120,
) -> list:
    """Broadcast a small ordered tx batch directly on Anvil.

    Anvil forks are prone to returning transient ``nonce too low`` errors when
    we feed already-signed sequential transactions through the multi-node retry
    broadcast helper that is designed for real RPC infrastructure. In unit
    tests we only have a single local Anvil node, so we re-sync the wallet
    nonce once against the local chain and then sign, submit and confirm the
    transactions one by one against the active provider.
    """
    provider = getattr(web3, "provider", None)
    active_provider = getattr(provider, "get_active_provider", lambda: provider)()
    direct_web3 = Web3(active_provider) if active_provider is not None else web3
    hot_wallet.sync_nonce(direct_web3)

    tx_hashes = []
    for bound_func, gas_limit in txs:
        signed_tx = hot_wallet.sign_bound_call_with_new_nonce(
            bound_func,
            tx_params={"gas": gas_limit},
            web3=direct_web3,
            fill_gas_price=True,
        )
        tx_hash = direct_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = direct_web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        if receipt["status"] != 1:
            assert_transaction_success_with_explanation(direct_web3, tx_hash)
        tx_hashes.append(tx_hash)

    return tx_hashes


class LagoonVaultSyncModel(AddressSyncModel):
    """Update Lagoon vault balances.

    - We do specific NAV update and settlement cycle to update
    """

    def __init__(
        self,
        vault: LagoonVault,
        hot_wallet: HotWallet | None,
        extra_gnosis_gas: int = 500_000,
        valuation_data_freshness=datetime.timedelta(hours=4),
        min_nav_change_update: Percent=0.005,
        unit_testing=False,
        calculate_valuation_func: Callable[..., USDollarPrice] | None = None,
        abort_lagoon_settlement_on_frozen_positions: bool = False,
    ):
        """
        :param extra_gnosis_gas:
            How much extra gas we need for transactions going through Gnosis machinery.

            Because of estimation problems.

        :param valuation_data_freshness:
            Crash is valuation data is older than this.

            Abort posting new valuations to onchain the valuation is too old.

        :param unit_testing:
            Don't use minor regorg safe latest block protection.

            Needed for tenderly.

        :param min_nav_change_update:
            Minimum change in NAV before we post update.

        :param calculate_valuation_func:
            Optional strategy-specific NAV calculation function.

            When set, :py:meth:`calculate_valuation` delegates to this
            callable instead of using the default
            ``portfolio.get_net_asset_value()``.

            This is needed for strategies where external systems (e.g.
            FreqTrade managing GMX positions) move funds in and out of
            the Safe without the trade engine knowing, making the
            portfolio's reserve balance stale.

            See :py:func:`tradeexecutor.exchange_account.gmx.create_gmx_vault_valuation_func`
            for the GMX-specific implementation.

        :param abort_lagoon_settlement_on_frozen_positions:
            Safety feature for live trading.

            When enabled, abort Lagoon settlement before NAV calculation if the
            strategy has frozen positions. This forces operators to resolve the
            frozen positions manually first and avoids miscounting or double
            counting capital in the posted NAV.
        """
        assert isinstance(vault, LagoonVault), f"Got {type(vault)} instead of LagoonVault"
        if hot_wallet is not None:
            # We can do initial setup without hot wallet
            assert isinstance(hot_wallet, HotWallet), f"Got {type(hot_wallet)} instead of HotWallet"
        self.vault = vault
        self.hot_wallet = hot_wallet
        self.extra_gnosis_gas = extra_gnosis_gas
        self.valuation_data_freshness = valuation_data_freshness
        self.min_nav_change_update = min_nav_change_update
        self.anvil = is_anvil(self.web3)  # Running test mode
        self.unit_testing = unit_testing  #
        self.calculate_valuation_func = calculate_valuation_func
        self.abort_lagoon_settlement_on_frozen_positions = abort_lagoon_settlement_on_frozen_positions
        assert vault.trading_strategy_module, "LagoonVault.trading_strategy_module initialisation param not set - needed to run the sync model properly"
        # assert isinstance(self.web3.provider, MEVBlockerProvider), f"This sync model needs MEVBlockerProvider, got {type(self.web3.provider)}"

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

    def has_async_deposits(self):
        return True

    def _check_frozen_positions_for_settlement(
        self,
        state: State,
        *,
        post_valuation: bool,
    ) -> None:
        """Abort Lagoon settlement before NAV calculation if frozen positions exist."""
        if not post_valuation:
            return

        if not self.abort_lagoon_settlement_on_frozen_positions:
            return

        frozen_count = len(state.portfolio.frozen_positions)
        if frozen_count == 0:
            return

        raise RuntimeError(
            "Lagoon settlement safety feature aborted settlement because the strategy has "
            f"{frozen_count} frozen position(s). Resolve frozen positions manually before "
            "calculating NAV to avoid miscounting or double counting capital."
        )

    def get_hot_wallet(self) -> Optional[HotWallet]:
        return self.hot_wallet

    def get_key_address(self) -> Optional[str]:
        return self.vault.vault_address

    def get_main_address(self) -> Optional[JSONHexAddress]:
        return self.vault.vault_address

    def get_token_storage_address(self) -> Optional[str]:
        return self.vault.safe_address

    def get_safe_latest_block(self) -> int:
        if self.anvil or self.unit_testing:
            # On Anvil tests, we need to always follow the latest block
            # Set self.unit_testing when using Tenderly
            return self.web3.eth.block_number
        else:
            # Leave room for minor reorg of 1-2 blocks
            return get_almost_latest_block_number(self.web3)

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

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> list[BalanceUpdate]:
        """Sync interests events.

        - Read interest gained onchain

        - Apply it to your state

        :return:
            The list of applied interest change events
        """

        return sync_interests(
            web3=self.web3,
            wallet_address=self.get_token_storage_address(),
            timestamp=timestamp,
            state=state,
            universe=universe,
            pricing_model=pricing_model,
        )

    def fetch_onchain_balances(
        self,
        assets: list[AssetIdentifier],
        filter_zero=True,
        block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:

        sorted_assets = sorted(assets, key=lambda a: a.address)

        # Latest block fails on LlamaNodes.com
        if block_identifier is None:
            block_identifier = self.get_safe_latest_block()

        return fetch_address_balances(
            self.web3,
            self.get_token_storage_address(),
            sorted_assets,
            block_number=block_identifier,
            filter_zero=filter_zero,
        )

    def calculate_valuation(self, state: State, *, block_number: int | None = None) -> USDollarPrice:
        """Calculate NAV of the vault.

        - Calculate the equity of all assets in the vault
        - Check that we do not use stale data
        - If a strategy-specific ``calculate_valuation_func`` was provided
          (e.g. for GMX), delegate to it; otherwise use the default
          ``portfolio.get_net_asset_value()``

        The freshness check always runs regardless of which valuation
        path is used.

        :param state:
            Current strategy state.
        :param block_number:
            Block number at which to read on-chain state.
            Forwarded to ``calculate_valuation_func`` when set.
        """

        now = native_datetime_utc_now()
        for p in state.portfolio.get_open_and_frozen_positions():
            if p.get_quantity() != 0:
                # Frozen positions may have quantity of 0 (failed open trades) and cannot have value
                valued_at = p.get_last_valued_at()
                updated_ago = now - valued_at
                last_event = p.valuation_updates[-1] if p.valuation_updates else None

                # Try to dump as much as possible information for diagnostics
                assert updated_ago < self.valuation_data_freshness, f"Position {p} pricing was too old for Lagoon valuation update. Now: {now}, updated at: {valued_at}, diff: {updated_ago}, threshold: {self.valuation_data_freshness}, last valuation event: {last_event}"

        if self.calculate_valuation_func is not None:
            return self.calculate_valuation_func(state, block_number=block_number)

        return state.portfolio.get_net_asset_value(include_interest=True)

    def _mark_treasury_sync_completed(
        self,
        treasury_sync,
        strategy_cycle_ts: datetime.datetime,
        block_number: int,
    ) -> None:
        """Mark Lagoon treasury as synced even when no settlement was needed."""
        treasury_sync.last_updated_at = native_datetime_utc_now()
        treasury_sync.last_cycle_at = strategy_cycle_ts
        treasury_sync.last_block_scanned = block_number

    def check_nav_update_and_settle_needed(self, calculated_nav: USDollarAmount) -> bool:
        """Do we need to settle or change onchain NAV.

        - Avoid unnecessary txs if NAV price has not moved

        :return:
            True if we need to settle/post NAV
        """
        block_number = self.get_safe_latest_block()
        flow_manager = self.vault.get_flow_manager()
        pending_deposits = flow_manager.fetch_pending_deposit(block_number)
        pending_redemptions = flow_manager.fetch_pending_redemption(block_number)
        if pending_deposits or pending_redemptions:
            logger.info("Deposit/redemptions detected, NAV update needed")
            return True

        onchain_nav = float(self.vault.fetch_nav(block_identifier=block_number))

        if onchain_nav > 0:
            nav_diff = abs(calculated_nav - onchain_nav) / onchain_nav
            if nav_diff >= self.min_nav_change_update:
                logger.info(
                    "NAV update needed. Calculated %s, onchain %s, diff %f %%",
                    calculated_nav,
                    onchain_nav,
                    nav_diff * 100,
                )
                return True
        else:
            # Avoid division by zero
            return not ((onchain_nav == 0) and (calculated_nav == 0))

        return False

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: Optional[list[AssetIdentifier]] = None,
        end_block: BlockNumber | NoneType = None,
        post_valuation=False,
    ) -> list[BalanceUpdate]:
        """Sync Lagoon treasury.

        - Calcualte NAV
        - Post it onchain if `post_valuation` is true
        - Will crash if the valuation or settle tx broadcast fails

        :param post_valuation:
            Doesn't do anything unless the post valuation is true.

            Because to get deposit events, we need to settle with a new valuation posted onchain.
        """

        web3 = self.web3
        sync = state.sync

        vault = self.vault
        treasury_sync = sync.treasury
        portfolio = state.portfolio

        assert sync.is_initialised(), f"Vault sync not initialised: {sync}\nPlease run trade-executor init command"

        match len(portfolio.reserves):
            case 1:
                # We have already run sync once
                logger.info("Reserve previously synced at %s", treasury_sync.last_updated_at)
                reserve_position = portfolio.get_default_reserve_position()
                reserve_asset = reserve_position.asset
            case 0:
                # Tabula rasa sync, need to create initial reserve position
                logger.info("Creating initial reserve")
                assert supported_reserves is not None
                reserve_asset = supported_reserves[0]
                state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
                reserve_position = portfolio.get_default_reserve_position()
            case _:
                raise NotImplementedError("Multireserve not supported")

        assert reserve_asset.is_stablecoin()

        reserve_token = fetch_erc20_details(
            web3,
            reserve_asset.address,
            cache=self.vault.token_cache,
            chain_id=reserve_asset.chain_id,
        )

        self._check_frozen_positions_for_settlement(
            state,
            post_valuation=post_valuation,
        )

        # Reconcile reserves from on-chain before calculating NAV.
        #
        # Exchange account positions (e.g. GMX) transfer USDC from the Safe
        # via sendTokens() in multicall — outside the trade engine.
        # This means reserve_position.quantity can be stale: it still
        # reflects the pre-transfer balance while the USDC has already
        # left the Safe.
        #
        # The exchange account value function (e.g. create_gmx_account_value_func)
        # only returns capital locked in exchange positions, NOT free USDC
        # in the Safe — so the Safe's actual USDC balance is the correct
        # reserve component for NAV.  Without this reconciliation,
        # calculate_valuation() double-counts the transferred USDC
        # (once in stale reserves, once in the exchange account position),
        # inflating the NAV and mispricing deposits.
        #
        # See README-GMX-Lagoon.md for the full token flow.
        block_number = self.get_safe_latest_block()
        onchain_balance = reserve_token.fetch_balance_of(
            self.get_token_storage_address(),
            block_identifier=block_number,
        )
        if reserve_position.quantity != onchain_balance:
            logger.warning(
                "Reserve balance mismatch: portfolio=%s, on-chain=%s. "
                "Updating to on-chain value before NAV calculation.",
                reserve_position.quantity,
                onchain_balance,
            )
            reserve_position.quantity = onchain_balance

        valuation = self.calculate_valuation(state, block_number=block_number)

        if not post_valuation:
            logger.warning("LagoonVaultSyncModel.sync_treasury() called with post_valuation=False")
            return []

        if not self.check_nav_update_and_settle_needed(valuation):
            self._mark_treasury_sync_completed(
                treasury_sync=treasury_sync,
                strategy_cycle_ts=strategy_cycle_ts,
                block_number=block_number,
            )
            logger.info("LagoonVaultSyncModel.sync_treasury() no actionable changes detected")
            return []

        assert self.hot_wallet, "asset_manager HotWallet needed in order to sync Lagoon vault"

        old_balance = reserve_token.fetch_balance_of(self.get_token_storage_address())

        logger.info("Posting new Lagoon valuation: %f USD", valuation)
        valuation_decimal = Decimal(valuation)
        valuation_func = vault.post_new_valuation(valuation_decimal)

        logger.info("Preparing to settle Lagoon")

        # Check if there's enough liquid USDC to cover pending redemptions
        block_number = web3.eth.block_number
        pending_shares = vault.get_flow_manager().fetch_pending_redemption(block_number)

        if pending_shares > 0:
            # Calculate how much USDC is needed for redemptions
            total_assets = vault.fetch_total_assets(block_number)
            total_supply = vault.fetch_total_supply(block_number)

            if total_supply > 0:
                share_price = total_assets / total_supply
                required_usdc = pending_shares * share_price

                # Check actual USDC balance in the Safe
                safe_usdc_balance = reserve_token.fetch_balance_of(vault.safe_address, block_number)

                logger.info(
                    "Redemption check: pending shares=%s, share price=%s, required USDC=%s, Safe balance=%s",
                    pending_shares,
                    share_price,
                    required_usdc,
                    safe_usdc_balance,
                )

                if required_usdc > safe_usdc_balance:
                    deficit = required_usdc - safe_usdc_balance
                    logger.warning(
                        "⚠️  INSUFFICIENT LIQUID USDC FOR REDEMPTIONS ⚠️\n"
                        "Pending redemptions: %s shares\n"
                        "Current share price: %s USDC/share\n"
                        "Required USDC: %s\n"
                        "Available in Safe: %s\n"
                        "Deficit: %s USDC\n"
                        "Redemptions will NOT be processed in this settlement cycle.\n"
                        "Consider redeeming from vault positions (IPOR/Morpho) before next settlement.",
                        pending_shares,
                        share_price,
                        required_usdc,
                        safe_usdc_balance,
                        deficit,
                    )

        settle_func = vault.settle_via_trading_strategy_module(valuation_decimal)

        if self.anvil or self.unit_testing:
            logger.info("Broadcasting Lagoon valuation + settle sequentially on Anvil")
            tx_hashes = _transact_anvil_sequentially(
                web3,
                self.hot_wallet,
                [
                    (valuation_func, DEFAULT_LAGOON_POST_VALUATION_GAS),
                    (settle_func, DEFAULT_LAGOON_SETTLE_GAS),
                ],
            )
            settle_tx_hash = tx_hashes[-1]
        else:
            signed_tx_1 = self.hot_wallet.sign_bound_call_with_new_nonce(
                valuation_func,
                tx_params={"gas": DEFAULT_LAGOON_POST_VALUATION_GAS},
                web3=web3,
                fill_gas_price=True
            )
            signed_tx_2 = self.hot_wallet.sign_bound_call_with_new_nonce(
                settle_func,
                tx_params={"gas": DEFAULT_LAGOON_SETTLE_GAS},
                web3=web3,
                fill_gas_price=True
            )
            wait_and_broadcast_multiple_nodes_mev_blocker(
                web3.provider,
                [signed_tx_1, signed_tx_2],
            )
            settle_tx_hash = signed_tx_2.hash

        analysis = analyse_vault_flow_in_settlement(
            vault,
            settle_tx_hash,
        )

        logger.info(
            "Lagoon settled. Settle result is:\n%s",
            pformat(analysis.get_serialiable_diagnostics_data())
        )

        # Post-settlement check: warn if redemptions were pending but not processed
        if analysis.pending_redemptions_shares > 0 and analysis.redeem_events == 0:
            logger.warning(
                "⚠️  REDEMPTIONS WERE NOT PROCESSED ⚠️\n"
                "Pending redemptions remain: %s shares (%s USDC)\n"
                "This typically indicates insufficient liquid USDC in the Safe.\n"
                "Redemption requests will remain pending until the next settlement cycle.",
                analysis.pending_redemptions_shares,
                analysis.pending_redemptions_underlying,
            )

        delta = analysis.get_underlying_diff()
        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        # Include our valuation in the other_data diangnostics
        other_data = analysis.get_serialiable_diagnostics_data()
        other_data["valuation"] = valuation
        valuation_with_deposits = valuation + float(delta)
        other_data["valuation_with_deposits"] = valuation_with_deposits

        share_count = vault.fetch_total_supply(analysis.block_number)
        other_data["share_count"] = share_count

        evt = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.reserve,
            cause=BalanceUpdateCause.deposit_and_redemption,
            asset=reserve_position.asset,
            block_mined_at=analysis.timestamp,
            strategy_cycle_included_at=strategy_cycle_ts,
            chain_id=reserve_asset.chain_id,
            old_balance=old_balance,
            quantity=delta,
            owner_address=None,
            tx_hash=analysis.tx_hash.hex(),
            log_index=None,
            position_id=None,
            usd_value=float(delta),  # Assume stablecoin
            notes=f"Lagoon reserve update at tx {analysis.tx_hash.hex()}, block {analysis.block_number:,}",
            block_number=analysis.block_number,
            other_data=other_data,
        )

        # Update reserve position mutable value
        reserve_position.reserve_token_price = float(1)
        reserve_position.last_pricing_at = analysis.timestamp
        reserve_position.last_sync_at = analysis.timestamp
        reserve_position.quantity = analysis.underlying_balance
        reserve_position.add_balance_update_event(evt)

        # Add in the event cross reference list
        ref = BalanceEventRef.from_balance_update_event(evt)
        treasury_sync.balance_update_refs.append(ref)
        treasury_sync.last_block_scanned = analysis.block_number
        treasury_sync.last_updated_at = native_datetime_utc_now()
        treasury_sync.last_cycle_at = strategy_cycle_ts
        treasury_sync.pending_redemptions = float(analysis.pending_redemptions_underlying)
        treasury_sync.share_count = share_count

        logger.info(
            f"Lagoon settlements done, the last block is now {treasury_sync.last_block_scanned:,}\n"
            f"Safe address: {vault.safe_address}, vault address: {vault.vault_address}, silo address: {vault.silo_address}\n"
            f"Settled {analysis.get_underlying_diff()} USD\n"
            f"Non-deposit valuation is {valuation:,.2f} USD, with-deposit valuation is {valuation_with_deposits:,.2f} USD\n"
            f"Pending redemptions {analysis.pending_redemptions_underlying} USD\n"
            f"Share count {share_count} {vault.share_token.symbol}"
        )
        return [evt]
