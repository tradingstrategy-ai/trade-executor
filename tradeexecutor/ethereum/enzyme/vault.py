"""Enzyme vaults integration."""

import logging
import datetime
import pprint
from _decimal import Decimal
from functools import partial
from types import NoneType
from typing import cast, List, Optional, Tuple, Iterable

from eth_defi.event_reader.conversion import convert_jsonrpc_value_to_int
from web3.types import BlockIdentifier

from eth_defi.provider.broken_provider import get_block_tip_latency, get_almost_latest_block_number
from web3 import Web3, HTTPProvider

from eth_defi.chain import fetch_block_timestamp, has_graphql_support
from eth_defi.enzyme.events import fetch_vault_balance_events, EnzymeBalanceEvent, Deposit, Redemption, fetch_vault_balances
from eth_defi.enzyme.vault import Vault
from eth_defi.event_reader.lazy_timestamp_reader import extract_timestamps_json_rpc_lazy, LazyTimestampContainer
from eth_defi.event_reader.reader import read_events, Web3EventReader, extract_events, extract_timestamps_json_rpc
from eth_defi.event_reader.reorganisation_monitor import ReorganisationMonitor
from eth_defi.hotwallet import HotWallet

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import BlockNumber
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


class EnzymeVaultSyncModel(SyncModel):
    """Update Enzyme vault balances."""

    def __init__(self,
                 web3: Web3,
                 vault_address: str,
                 reorg_mon: ReorganisationMonitor,
                 only_chain_listener=True,
                 hot_wallet: Optional[HotWallet] = None,
                 generic_adapter_address: Optional[str] = None,
                 vault_payment_forwarder_address: Optional[str] = None,
                 scan_chunk_size=10_000,
                 ):
        """

        :param web3:
            Web3

        :param vault_address:
            The address of the vault

        :param reorg_mon:
            How to deal with block updates

        :param only_chain_listerer:
            This is the only adapter using reorg_mon.

            Will call :py:meth:`process_blocks` as the part :py:meth:`sync_treasury`.

        :param hot_wallet:
            Trade executor's hot wallet used to create transactions.

            Only needed when doing trades.

        :param generic_adapter_address:
            The vault specific deployed GenericAdapter smart contract.

            Needed to make trades.

        :param scan_chunk_size:
            Ethereum eth_getLogs JSON-RPC workaround for a horrible blockchain APIs.
        """
        assert vault_address is not None, "Vault address is not given"
        self.web3 = web3
        self.reorg_mon = reorg_mon
        try:

            self.vault = Vault.fetch(
                web3,
                vault_address,
                generic_adapter_address,
                payment_forwarder=vault_payment_forwarder_address,
            )
        except Exception as e:
            raise RuntimeError(f"Could not fetch Enzyme vault data for {vault_address}: {e}") from e
        self.scan_chunk_size = scan_chunk_size
        self.only_chain_listener = only_chain_listener
        self.hot_wallet = hot_wallet

    def __repr__(self):
        return f"<EnzymeVaultSyncModel for vault {self.vault.address} using hot wallet {self.hot_wallet.address if self.hot_wallet else '(not set)'}>"

    def resync_nonce(self):
        self.hot_wallet.sync_nonce(self.web3)

    def get_vault_address(self) -> Optional[str]:
        """Get the vault address we are using"""
        return self.vault.address

    def get_hot_wallet(self) -> Optional[HotWallet]:
        return self.hot_wallet

    def get_token_storage_address(self) -> Optional[str]:
        return self.get_vault_address()

    def is_ready_for_live_trading(self, state: State) -> bool:
        """Have we run init command on the vault."""
        return state.sync.deployment.block_number is not None

    def _notify(
            self,
            current_block: int,
            start_block: int,
            end_block: int,
            chunk_size: int,
            total_events: int,
            last_timestamp: int,
            context,
    ):
        """Log notifier used in Enzyme event reading"""
        # Because the code is only run once ever,
        # we show the progress by printing
        if end_block - start_block > 0:
            done = (current_block - start_block) / (end_block - start_block)
            logger.info(f"EnzymeVaultSyncMode: Scanning blocks {current_block:,} - {current_block + chunk_size:,}, done {done * 100:.1f}%")

    def process_blocks(self) -> Tuple[int, int]:
        """Process the reorgsanisation monitor blocks.

        :raise ChainReorganisationDetected:
            When any if the block data in our internal buffer
            does not match those provided by events.

        :return:
            Range to scan for the events
        """
        range_start = self.reorg_mon.last_block_read
        reorg_resolution = self.reorg_mon.update_chain()
        range_end = self.reorg_mon.last_block_read
        return reorg_resolution.get_read_range()

    def fetch_vault_reserve_asset(self) -> AssetIdentifier:
        """Read the reserve asset from the vault data."""
        token = self.vault.denomination_token
        address = token.address
        assert type(address) == str
        return translate_token_details(token)

    def get_related_position(self, portfolio: Portfolio, asset: AssetIdentifier) -> ReservePosition | TradingPosition:
        """Map a redemption event asset to an underlying position.

        :raise UnknownAsset:
            If we got a redemption event for an asset that does not belong to any of our positions

        """
        assert len(portfolio.reserves) == 1, "Safety check"
        reserve_position = portfolio.reserves.get(asset.address)
        if reserve_position is not None:
            return reserve_position

        spot_position = portfolio.get_open_position_for_asset(asset)
        if spot_position:
            return spot_position

        position_str = ", ".join([str(p) for p in portfolio.get_open_positions()])
        raise UnknownAsset(f"Asset {asset} does not map to any open position.\n"
                           f"Reserve: {portfolio.get_default_reserve_asset()}.\n"
                           f"Open positions: {position_str}")

    def process_deposit(self, portfolio: Portfolio, event: Deposit, strategy_cycle_ts: datetime.datetime) -> BalanceUpdate:
        """Translate Enzyme SharesBought event to our internal deposit storage format."""

        asset = translate_token_details(event.denomination_token)
        if len(portfolio.reserves) == 0:
            # Initial deposit
            portfolio.initialise_reserves(asset)
        else:
            reserve_asset, reserve_price = portfolio.get_default_reserve_asset()
            assert asset == reserve_asset

        reserve_position = portfolio.get_reserve_position(asset)
        old_balance = reserve_position.quantity
        exchange_rate = self.vault.fetch_denomination_token_usd_exchange_rate()
        reserve_position.reserve_token_price = float(exchange_rate)
        reserve_position.last_pricing_at = datetime.datetime.utcnow()
        reserve_position.last_sync_at = datetime.datetime.utcnow()
        reserve_position.quantity += event.investment_amount

        usd_value = float(exchange_rate) * float(event.investment_amount)

        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        assert event.timestamp is not None, f"Timestamp cannot be none: {event}"

        evt = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.reserve,
            cause=BalanceUpdateCause.deposit,
            asset=asset,
            block_mined_at=event.timestamp,
            strategy_cycle_included_at=strategy_cycle_ts,
            chain_id=asset.chain_id,
            old_balance=old_balance,
            usd_value=usd_value,
            quantity=event.investment_amount,
            owner_address=event.receiver,
            tx_hash=event.event_data["transactionHash"],
            log_index=event.event_data["logIndex"],
            position_id=None,
            block_number=convert_jsonrpc_value_to_int(event.event_data["blockNumber"]),
        )

        reserve_position.add_balance_update_event(evt)

        return evt

    def process_redemption(self, portfolio: Portfolio, event: Redemption, strategy_cycle_ts: datetime.datetime) -> List[BalanceUpdate]:
        """Translate Enzyme SharesBought event to our internal deposit storage format.

        In-kind redemption exchanges user share tokens to underlying
        assets.

        - User gets whatever strategy reserves there is

        - User gets share of whatever spot positions there are currently open
        """

        events = []

        for token_details, raw_amount in event.redeemed_assets:

            if raw_amount == 0:
                # Enzyme reports zero redemptions for some reason?
                # enzyme-polygon-matic-usdc  | <USD Coin (PoS) (USDC) at 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174, 6 decimals, on chain 137>: 999324
                # enzyme-polygon-matic-usdc  | <Wrapped Matic (WMATIC) at 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270, 18 decimals, on chain 137>: 0
                continue

            asset = translate_token_details(token_details)

            try:
                position = self.get_related_position(portfolio, asset)
            except UnknownAsset as e:
                # Something has gone wrong in accounting, as we cannot match the redeemed asset to any open position.
                # This is very tricky situation to figure out, so we be verbose with error messages.
                open_positions = "\n".join([str(p) for p in portfolio.get_open_positions()])
                assets_msg = "\n".join([f"{r}: {amount}" for r, amount in event.redeemed_assets])
                msg = f"Redemption failure because redeemed asset does not match our internal accounting.\n" \
                      f"Do not know how to recover. You need to stop trade-executor and run accounting correction.\n" \
                      f"Could not process redemption event.\n" \
                      f"Redeemed assets:\n" \
                      f"{assets_msg}\n" \
                      f"EVM event data:\n" \
                      f"{_dump_enzyme_event(event)}\n" \
                      f"Open positions currently in the state:\n" \
                      f"{open_positions or '-'}"
                raise RedemptionFailure(msg) from e

            quantity = asset.convert_to_decimal(raw_amount)

            assert quantity > 0  # Sign flipped later

            event_id = portfolio.next_balance_update_id
            portfolio.next_balance_update_id += 1

            if isinstance(position, ReservePosition):
                position_id = None
                old_balance = position.quantity
                position.quantity -= quantity
                position_type = BalanceUpdatePositionType.reserve
                # TODO: USD stablecoin hardcoded to 1:1 with USD
                usd_value = float(quantity)
            elif isinstance(position, TradingPosition):
                position_id = position.position_id
                old_balance = position.get_quantity()
                position_type = BalanceUpdatePositionType.open_position
                usd_value = position.calculate_quantity_usd_value(quantity)
            else:
                raise NotImplementedError()

            assert old_balance - quantity >= 0, f"Position went to negative: {position} with token {token_details} and amount {raw_amount}\n" \
                                                f"Quantity: {quantity}, old balance: {old_balance}"

            assert event.timestamp is not None, f"Timestamp cannot be none: {event}"

            evt = BalanceUpdate(
                balance_update_id=event_id,
                cause=BalanceUpdateCause.redemption,
                position_type=position_type,
                asset=asset,
                block_mined_at=event.timestamp,
                strategy_cycle_included_at=strategy_cycle_ts,
                chain_id=asset.chain_id,
                quantity=-quantity,
                old_balance=old_balance,
                owner_address=event.redeemer,
                tx_hash=event.event_data["transactionHash"],
                log_index=event.event_data["logIndex"],
                position_id=position_id,
                usd_value=usd_value,
                block_number=convert_jsonrpc_value_to_int(event.event_data["blockNumber"]),
            )

            position.add_balance_update_event(evt)

            events.append(evt)

            if isinstance(position, TradingPosition) and position.is_credit_supply():
                update_credit_supply_loan(
                    loan=position.loan,
                    position=position,
                    quantity_delta=-quantity,
                    timestamp=event.timestamp,
                    mode="execute",
                )

        return events

    def translate_and_apply_event(self, state: State, event: EnzymeBalanceEvent, strategy_cycle_ts: datetime.datetime) -> List[BalanceUpdate]:
        """Translate on-chain event data to our persistent format."""
        portfolio = state.portfolio

        match event:
            case Deposit():
                # Deposit generated only one event
                event = cast(Deposit, event)
                logger.info("Processing Enzyme deposit %s", event.event_data)
                return [self.process_deposit(portfolio, event, strategy_cycle_ts)]
            case Redemption():
                # Enzyme in-kind redemption can generate updates for multiple assets
                event = cast(Redemption, event)
                logger.info("Processing Enzyme redemption %s", event.event_data)
                # Sanity check: Make sure there has not been redemptions from the vault before the strategy was initialised.
                # Make sure we do not get events that are from the time before
                # the state was initialised
                first_allowed_ts = state.sync.deployment.initialised_at
                if first_allowed_ts is not None:
                    assert event.timestamp > first_allowed_ts, f"Vault has a redemption from the time before trade execution was initialised\n" \
                                                               f"Initialised at: {state.sync.deployment.initialised_at}\n" \
                                                               f"Event:\n" \
                                                               f"{_dump_enzyme_event(event)}" \

                return self.process_redemption(portfolio, event, strategy_cycle_ts)
            case _:
                raise RuntimeError(f"Unsupported event: {event}")

    def sync_initial(self, state: State, allow_override=False, **kwargs):
        """Get the deployment event by scanning the whole chain from the start.

        Updates `state.sync.deployment` structure.

        .. note::

            You need to give `start_block` hint of the scanning will take too long because
            Ethereum design flaws.

        Example:

        .. code-block:: python

            sync_model.sync_initial(state, start_block=35_123_123)

        """
        sync = state.sync


        if not allow_override:
            assert not sync.is_initialised(), "Initialisation twice is not allowed"

        web3 = self.web3
        deployment = state.sync.deployment

        # Set up the reader interface for fetch_deployment_event()
        # extract_timestamp is disabled to speed up the event reading,
        # we handle it separately
        reader: Web3EventReader = cast(
            Web3EventReader,
            partial(read_events, notify=self._notify, chunk_size=self.scan_chunk_size, extract_timestamps=None)
        )

        start_block = kwargs.get("start_block")
        if not start_block:
            start_block = 1

        logger.info("Starting event scan at the block %d, chunk size is %d", start_block, self.scan_chunk_size)
        deployment_event = self.vault.fetch_deployment_event(reader=reader, start_block=start_block)

        # Check that we got good event data
        block_number = deployment_event["blockNumber"]
        block_hash = deployment_event["blockHash"]
        tx_hash = deployment_event["transactionHash"]
        assert block_number > 1

        # Get the block info to get the timestamp for the event
        block_data = extract_timestamps_json_rpc(web3, block_number, block_number)
        timestamp_unix = block_data[block_hash]
        timestamp_dt = datetime.datetime.utcfromtimestamp(timestamp_unix)

        deployment.address = self.vault.vault.address
        deployment.block_number = block_number
        deployment.tx_hash = tx_hash
        deployment.block_mined_at = timestamp_dt
        deployment.vault_token_name = self.vault.get_name()
        deployment.vault_token_symbol = self.vault.get_symbol()
        deployment.chain_id = ChainId(web3.eth.chain_id)

        current_block = web3.eth.block_number
        deployment.initialised_at = fetch_block_timestamp(web3, current_block)

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> List[BalanceUpdate]:
        """Sync interests events.

        - Read interest gained onchain

        - Apply it to your state

        :return:
            The list of applied interest change events
        """

        return sync_interests(
            web3=self.web3,
            wallet_address=self.get_vault_address(),
            timestamp=timestamp,
            state=state,
            universe=universe,
            pricing_model=pricing_model,
        )

    def fetch_onchain_balances(
            self,
            assets: List[AssetIdentifier],
            filter_zero=True,
            block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:

        sorted_assets = sorted(assets, key=lambda a: a.address)

        # Latest block fails on LlamaNodes.com
        if block_identifier is None:
            block_identifier = get_almost_latest_block_number(self.web3)

        return fetch_address_balances(
            self.web3,
            self.get_vault_address(),
            sorted_assets,
            block_number=block_identifier,
            filter_zero=filter_zero,
        )

    def create_event_reader(self) -> Tuple[Web3EventReader, bool]:
        """Create event reader for vault deposit/redemption events.

        Set up the reader interface for fetch_deployment_event()
        extract_timestamp is disabled to speed up the event reading,
        we handle it separately

        :return:
            Tuple (event reader, quick node workarounds).
        """

        # TODO: make this a configuration option
        provider = cast(HTTPProvider, self.web3.provider)
        if has_graphql_support(provider):
            logger.info("Using /graqpql based reader for vault events")
            # GoEthereum with /graphql enabled
            reader: Web3EventReader = cast(
                Web3EventReader,
                partial(read_events, notify=self._notify, chunk_size=self.scan_chunk_size, reorg_mon=self.reorg_mon, extract_timestamps=None)
            )

            broken_quicknode = False
        else:
            # Fall back to lazy load event timestamps,
            # all commercial SaaS nodes

            # QuickNode is crap
            # We do not explicitly detect QuickNode, but we are getting
            # equests.exceptions.HTTPError: 413 Client Error: Request Entity Too Large for url: https://xxx.pro/ec618a382930d83cdbeb0119eae1694c480ce789/
            chunk_size = 1000

            logger.info("Using lazy timestamp loading reader for vault events")
            lazy_timestamp_container: LazyTimestampContainer = None

            def wrapper(web3, start_block, end_block):
                nonlocal lazy_timestamp_container
                lazy_timestamp_container = extract_timestamps_json_rpc_lazy(web3, start_block, end_block)
                return lazy_timestamp_container

            logger.info("Using lazy timestamp loading reader for vault events")
            reader: Web3EventReader = cast(
                Web3EventReader,
                partial(read_events, notify=self._notify, chunk_size=chunk_size, reorg_mon=None, extract_timestamps=wrapper))

            if lazy_timestamp_container:
                logger.info("Made %d eth_getBlockByNumber API calls", lazy_timestamp_container.api_call_counter)
            else:
                logger.info("Event reader not called")

            broken_quicknode = True

        return reader, broken_quicknode

    def sync_treasury(self,
                      strategy_cycle_ts: datetime.datetime,
                      state: State,
                      supported_reserves: Optional[List[AssetIdentifier]] = None,
                      end_block: BlockNumber | NoneType = None,
                      ) -> List[BalanceUpdate]:

        web3 = self.web3
        sync = state.sync
        assert sync.is_initialised(), f"Vault sync not initialised: {sync}\nPlease run trade-executor init command"

        vault = self.vault

        treasury_sync = sync.treasury

        if treasury_sync.last_block_scanned:
            start_block = treasury_sync.last_block_scanned + 1
        else:
            start_block = sync.deployment.block_number

        web3 = self.web3

        if not end_block:
            # Legacy
            end_block = get_almost_latest_block_number(web3)

        last_block = web3.eth.block_number

        logger.info(f"Starting treasury sync for vault {self.vault.address}, comptroller {self.vault.comptroller.address}, looking block range {start_block:,} - {end_block:,}, last block is {last_block:,}")

        reader, broken_quicknode = self.create_event_reader()

        # Feed block headers for the listeners
        # to get the timestamps of the blocks
        if self.only_chain_listener and not broken_quicknode:

            skip_to_block = treasury_sync.last_block_scanned or sync.deployment.block_number

            known_block_count = len(self.reorg_mon.block_map)

            if not known_block_count:
                logger.info("Loading initial block data, skipping reorg mon to to block %s, reorg mon has %d entries", skip_to_block, known_block_count)
                range_start = max(skip_to_block - self.reorg_mon.check_depth, 1)
                self.reorg_mon.load_initial_block_headers(start_block=range_start)
                range_end = self.reorg_mon.last_block_read
            else:
                logger.info("Loading more block data, reorg mon has %d entries", known_block_count)
                range_start, range_end = self.process_blocks()

            logger.info("Reorg mon has %d block headers", len(self.reorg_mon.block_map))
            treasury_sync.last_block_scanned = range_end
        else:
            range_start = start_block
            range_end = end_block

        events_iter = fetch_vault_balance_events(
            vault,
            start_block=range_start,
            end_block=range_end,
            read_events=reader,
        )

        events = []
        for chain_event in events_iter:
            events += self.translate_and_apply_event(state, chain_event, strategy_cycle_ts)
            for e in events:
                logger.info(f"Processed Enzyme balance update event %s", e)

        # Check that we do not have conflicting events
        new_event: BalanceUpdate
        for new_event in events:
            ref = BalanceEventRef(
                balance_event_id=new_event.balance_update_id,
                strategy_cycle_included_at=new_event.strategy_cycle_included_at,
                cause=new_event.cause,
                position_type=new_event.position_type,
                position_id=new_event.position_id,
                usd_value=new_event.usd_value,
            )
            treasury_sync.balance_update_refs.append(ref)

        treasury_sync.last_block_scanned = end_block
        treasury_sync.last_updated_at = datetime.datetime.utcnow()
        treasury_sync.last_cycle_at = strategy_cycle_ts

        # Update the reserve position value
        # TODO: Add USDC/USD price feed
        # state.portfolio.get_default_reserve_position().update_value(exchange_rate=1.0)

        logger.info(f"Enzyme treasury sync done, the last block is now {treasury_sync.last_block_scanned:,}, found {len(events)} events")

        return events

    def sync_reinit(self, state: State, allow_override=False, **kwargs):
        """Reinitiliase the vault.

        Fixes broken accounting. Only needs to be used if internal state and blockchain
        state have somehow managed to get out of sync: internal state has closed positions
        that are not in blockchain state or vice versa.

        - Makes any token balances in the internal state to match the blockchain state.

        - Assumes all positions are closed (currently artificial limitation).

        - All position history is deleted, because we do not know whether positions closed for
          profit or loss.

        See :py:mod:`tradeexexcutor.cli.commands.reinit` for details.

        .. note::

            Currently quite a test code. Make support all positions, different sync models.

        :param state:
            Empty state

        :param allow_override:
            Allow init twice.

        :param kwargs:
            Initial sync hints.

            Passed to :py:meth:`sync_initial`.
        """

        # First set the vault creation date etc.
        self.sync_initial(state, allow_override=allow_override, **kwargs)

        # Then proceed to construct the balacnes from the EVM state
        web3 = self.web3
        vault = self.vault
        sync = state.sync
        portfolio = state.portfolio
        treasury_sync = sync.treasury

        current_block = web3.eth.block_number
        timestamp = fetch_block_timestamp(web3, current_block)

        # Get all non-zero balances from Enzyme
        balances = [b for b in fetch_vault_balances(vault, block_identifier=current_block) if b.balance > 0]

        assert len(balances) == 1, f"reinit cannot be done if the vault has positions other than reserve currencies, got {balances}"
        reserve_current_balance = balances[0]

        logger.info("Found on-chain balance %s at block %s", reserve_current_balance, current_block)

        asset = translate_token_details(reserve_current_balance.token)

        assert len(portfolio.reserves) == 0

        # Initial deposit
        portfolio.initialise_reserves(asset)

        reserve_position = portfolio.get_reserve_position(asset)

        reserve_position.reserve_token_price = float(1)
        reserve_position.last_pricing_at = datetime.datetime.utcnow()
        reserve_position.last_sync_at = datetime.datetime.utcnow()
        reserve_position.quantity = reserve_current_balance.balance

        # TODO: Assume USD stablecoins are 1:1 USD
        usd_value = float(reserve_position.quantity)

        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        master_event = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.reserve,
            cause=BalanceUpdateCause.deposit,
            asset=asset,
            block_mined_at=timestamp,
            strategy_cycle_included_at=timestamp,
            chain_id=asset.chain_id,
            old_balance=Decimal(0),
            quantity=reserve_current_balance.balance,
            owner_address=None,
            tx_hash=None,
            log_index=None,
            position_id=None,
            usd_value=usd_value,
            notes=f"reinit() at block {current_block}"
        )

        reserve_position.add_balance_update_event(master_event)

        events = [master_event]

        # Check that we do not have conflicting events
        new_event: BalanceUpdate
        for new_event in events:
            ref = BalanceEventRef(
                balance_event_id=new_event.balance_update_id,
                strategy_cycle_included_at=new_event.strategy_cycle_included_at,
                cause=new_event.cause,
                position_type=new_event.position_type,
                position_id=new_event.position_id,
                usd_value=new_event.usd_value,
            )
            treasury_sync.balance_update_refs.append(ref)

        treasury_sync.last_block_scanned = current_block
        treasury_sync.last_updated_at = datetime.datetime.utcnow()
        treasury_sync.last_cycle_at = None

    def create_transaction_builder(self) -> EnzymeTransactionBuilder:
        assert self.hot_wallet, "HotWallet not set - cannot create transaction builder"
        return EnzymeTransactionBuilder(self.hot_wallet, self.vault)

    def check_ownership(self):
        """Check that the hot wallet has the correct ownership rights to make trades through the vault.

        Hot wallet must be registered either as

        - Vault owner

        - Vault asset manager

        :raise AssertionError:
            If the hot wallet cannot perform trades for the vault
        """

        # TODO: Move this check internal on EnzymeVaultSyncModel
        hot_wallet = self.get_hot_wallet()
        vault = self.vault

        owner = vault.vault.functions.getOwner().call()
        if owner != hot_wallet.address:
            assert vault.vault.functions.isAssetManager(hot_wallet.address).call(), f"Address is not set up as Enzyme asset manager: {hot_wallet.address}, owner is {owner}, isAssetManager() returns false"

    def reset_deposits(self, state: State):
        """Clear out pending withdrawals/deposits events."""
        web3 = self.web3
        current_block = web3.eth.block_number

        # Skip all deposit/redemption events between the last scanned block and now
        sync = state.sync
        treasury_sync = sync.treasury
        treasury_sync.last_block_scanned = current_block
        treasury_sync.last_updated_at = datetime.datetime.utcnow()
        treasury_sync.last_cycle_at = None

def _dump_enzyme_event(e: EnzymeBalanceEvent) -> str:
    """Format enzyme events in the error / log output."""
    # Dump internal JSON-RPC JSON
    return pprint.pformat(e.event_data, indent=2)