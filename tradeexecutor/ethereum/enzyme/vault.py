"""Enzyme vaults integration."""


import logging
import dataclasses
import datetime
from decimal import Decimal
from functools import partial
from typing import Dict, List, cast

from dataclasses_json import dataclass_json
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.balances import DecimalisedHolding, \
    fetch_erc20_balances_by_token_list, convert_balances_to_decimal
from eth_defi.enzyme.vault import Vault
from eth_defi.event_reader.reader import read_events, Web3EventReader, extract_events, extract_timestamps_json_rpc
from tradeexecutor.state.reserve import ReservePosition

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.sync_model import SyncModel

logger = logging.getLogger(__name__)


@dataclass_json
@dataclasses.dataclass
class ReserveUpdateEvent:
    asset: AssetIdentifier
    updated_at: datetime.datetime
    past_balance: Decimal
    new_balance: Decimal


def update_wallet_balances(web3: Web3, address: HexAddress, tokens: List[HexAddress]) -> Dict[HexAddress, DecimalisedHolding]:
    """Get raw balances of ERC-20 tokens."""
    balances = fetch_erc20_balances_by_token_list(web3, address, tokens)
    return convert_balances_to_decimal(web3, balances)


def sync_reserves(
        web3: Web3,
        clock: datetime.datetime,
        wallet_address: HexAddress,
        current_reserves: List[ReservePosition],
        supported_reserve_currencies: List[AssetIdentifier]) -> List[ReserveUpdateEvent]:
    """Check the address for any incoming stablecoin transfers to see how much cash we have."""

    our_chain_id = web3.eth.chain_id

    # Get raw ERC-20 holdings of the address
    balances = update_wallet_balances(web3, wallet_address, [web3.toChecksumAddress(a.address) for a in supported_reserve_currencies])

    reserves_per_token = {r.asset.address: r for r in current_reserves}

    events: ReserveUpdateEvent = []

    for currency in supported_reserve_currencies:

        address = currency.address

        # 1337 is Ganache
        if our_chain_id != 1337:
            assert currency.chain_id == our_chain_id, f"Asset expects chain_id {currency.chain_id}, currently connected to {our_chain_id}"

        if currency.address in reserves_per_token:
            # We have an existing record of having this reserve
            current_value = reserves_per_token[address].quantity
        else:
            current_value = Decimal(0)

        decimal_holding = balances.get(Web3.toChecksumAddress(address))

        # We get decimals = None if Ganache is acting
        assert decimal_holding.decimals, f"Token did not have decimals: token:{currency} holding:{decimal_holding}"

        if (decimal_holding is not None) and (decimal_holding.value != current_value):
            evt = ReserveUpdateEvent(
                asset=currency,
                past_balance=current_value,
                new_balance=decimal_holding.value,
                updated_at=clock
            )
            events.append(evt)
            logger.info("Reserve currency update detected. Asset: %s, past: %s, new: %s", evt.asset, evt.past_balance, evt.new_balance)

    return events


def sync_balances(
        web3: Web3,
        clock: datetime.datetime,
        wallet_address: HexAddress,
        current_reserves: List[ReservePosition],
        supported_reserve_currencies: List[AssetIdentifier]) -> List[ReserveUpdateEvent]:
    """Sync Enzyme vault balances.

    Enzyme vault can have

    - Deposits

    - In-kind redemptions
    """

    our_chain_id = web3.eth.chain_id

    # Get raw ERC-20 holdings of the address
    balances = update_wallet_balances(web3, wallet_address, [web3.toChecksumAddress(a.address) for a in supported_reserve_currencies])

    reserves_per_token = {r.asset.address: r for r in current_reserves}

    events: ReserveUpdateEvent = []

    for currency in supported_reserve_currencies:

        address = currency.address

        # 1337 is Ganache
        if our_chain_id != 1337:
            assert currency.chain_id == our_chain_id, f"Asset expects chain_id {currency.chain_id}, currently connected to {our_chain_id}"

        if currency.address in reserves_per_token:
            # We have an existing record of having this reserve
            current_value = reserves_per_token[address].quantity
        else:
            current_value = Decimal(0)

        decimal_holding = balances.get(Web3.toChecksumAddress(address))

        # We get decimals = None if Ganache is acting
        assert decimal_holding.decimals, f"Token did not have decimals: token:{currency} holding:{decimal_holding}"

        if (decimal_holding is not None) and (decimal_holding.value != current_value):
            evt = ReserveUpdateEvent(
                asset=currency,
                past_balance=current_value,
                new_balance=decimal_holding.value,
                updated_at=clock
            )
            events.append(evt)
            logger.info("Reserve currency update detected. Asset: %s, past: %s, new: %s", evt.asset, evt.past_balance, evt.new_balance)

    return events


class EnzymeVaultSyncModel(SyncModel):
    """Update Enzyme vault balances."""

    def __init__(self,
                 web3: Web3,
                 vault_address: str,
                 ):
        self.web3 = web3
        self.vault = Vault.fetch(web3, vault_address)

    def sync_initial(self, state: State):
        """Get the deployment event by scanning the whole chain from the start"""
        sync = state.sync
        assert not sync.is_initialised(), "Initialisation twice is not allowed"

        web3 = self.web3
        deployment = state.sync.deployment

        def notify(
            current_block: int,
            start_block: int,
            end_block: int,
            chunk_size: int,
            total_events: int,
            last_timestamp: int,
            context,
        ):
            # Because the code is only run once ever,
            # we show the progress by
            done = (current_block - start_block) / (end_block - start_block)
            logger.info(f"EnzymeVaultSyncModel.sync_initial(): Scanning blocks {current_block:,} - {current_block + chunk_size:,}, done {done * 100:.1f}%")

        # Set up the reader interface for fetch_deployment_event()
        # extract_timestamp is disabled to speed up the event reading,
        # we handle it separately
        reader: Web3EventReader = cast(Web3EventReader, partial(read_events, notify=notify, chunk_size=10_000, extract_timestamps=None))

        deployment_event = self.vault.fetch_deployment_event(reader=extract_events)

        # Check that we got good event data
        block_number = deployment_event["blockNumber"]
        block_hash = deployment_event["blockHash"]
        tx_hash = deployment_event["transactionHash"]
        assert block_number > 1

        # Get the block info to get the timestamp for the event
        block_data =  extract_timestamps_json_rpc(web3, block_number, block_number)
        timestamp_unix = block_data[block_hash]
        timestamp_dt = datetime.datetime.utcfromtimestamp(timestamp_unix)

        deployment.address = self.vault.vault.address
        deployment.block_number = block_number
        deployment.tx_hash = tx_hash
        deployment.block_mined_at = timestamp_dt
        deployment.vault_token_name = self.vault.get_name()
        deployment.vault_token_symbol = self.vault.get_symbol()

    def sync_treasury(self,
                 strategy_cycle_ts: datetime.datetime,
                 state: State,
                 ):
        """Apply the balance sync before each strategy cycle."""
        pass
