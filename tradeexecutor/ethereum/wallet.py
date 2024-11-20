"""Deposit and withdraw detection and management."""

import logging
import dataclasses
import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json
from eth_typing import HexAddress
from web3 import Web3
from web3.types import BlockIdentifier

from eth_defi.balances import DecimalisedHolding, \
    fetch_erc20_balances_by_token_list, convert_balances_to_decimal
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.run_state import RunState

logger = logging.getLogger(__name__)


@dataclass_json
@dataclasses.dataclass
class ReserveUpdateEvent:
    """A legacy reserve update event.

    Maintained for old code compatibility.

    See :py:mod:`tradeeexecutor.state.sync` for the current approach.

    TODO: This should be removed as is partially part of old treasury sync code.
    """
    asset: AssetIdentifier

    #: Transfer timestamp (if known)
    mined_at: datetime.datetime

    #: Strategy cycle timestamp
    updated_at: datetime.datetime
    past_balance: Decimal
    new_balance: Decimal

    block_number: int | None = None

    @property
    def change(self) -> Decimal:
        return self.new_balance - self.past_balance


def update_wallet_balances(
        web3: Web3,
        address: HexAddress,
        tokens: List[HexAddress],
        block_identifier: BlockIdentifier = None,
) -> Dict[HexAddress, DecimalisedHolding]:
    """Get raw balances of ERC-20 tokens."""

    if block_identifier is None:
        block_identifier = get_almost_latest_block_number(web3)

    balances = fetch_erc20_balances_by_token_list(web3, address, tokens, block_identifier=block_identifier)
    return convert_balances_to_decimal(web3, balances)


def sync_reserves(
    web3: Web3,
    clock: datetime.datetime,
    wallet_address: HexAddress,
    current_reserves: List[ReservePosition],
    supported_reserve_currencies: List[AssetIdentifier],
    block_identifier: BlockIdentifier = None,
) -> List[ReserveUpdateEvent]:
    """Check the address for any incoming stablecoin transfers to see how much cash we have."""

    assert supported_reserve_currencies, f"Supported reserve currency address empty when syncing: {wallet_address}"

    our_chain_id = web3.eth.chain_id

    if block_identifier is None:
        block_identifier = get_almost_latest_block_number(web3)

    # Get raw ERC-20 holdings of the address
    balances = update_wallet_balances(
        web3,
        wallet_address,
        [web3.to_checksum_address(a.address) for a in supported_reserve_currencies],
        block_identifier=block_identifier,
    )

    # Make sure we avoid checksummed string addresses from now on
    balances = {k.lower(): v for k,v in balances.items()}

    reserves_per_token = {r.asset.address.lower(): r for r in current_reserves}

    events: List[ReserveUpdateEvent] = []

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

        decimal_holding = balances.get(address)

        # We get decimals = None if Ganache is acting
        assert decimal_holding.decimals, f"Token did not have decimals: token:{currency} holding:{decimal_holding}"

        if (decimal_holding is not None) and (decimal_holding.value != current_value):
            evt = ReserveUpdateEvent(
                asset=currency,
                past_balance=current_value,
                new_balance=decimal_holding.value,
                updated_at=clock,
                mined_at=clock,  # TODO: We do not have logic to get actual block_mined_at of Transfer() here
                block_number=block_identifier,
            )
            events.append(evt)
            logger.info("Reserve currency update detected. Asset: %s, past: %s, new: %s", evt.asset, evt.past_balance, evt.new_balance)

    return events


def perform_gas_level_checks(
    web3: Web3,
    run_state: RunState,
    hot_wallet: HotWallet,
    hot_wallet_gas_warning_level: Optional[Decimal] = None,
) -> bool:
    """Check the gas level of the hot wallet.

    - We need gas money to perform transactions

    - Issue a low gas warnign if gas is running low

    - Print a log message

    - Set :py:class:`RunState`` flag.

    - Does not do anything if warning gas level is not set

    - Clear the existing warning, if the gas warning light is on

    :param hot_wallet_gas_warning_level:
        Set this or one in ``RunState.hot_wallet_gas_warning_level``.

    :return:
        True if the gas warning is going
    """

    hot_wallet_gas_warning_level = hot_wallet_gas_warning_level or run_state.hot_wallet_gas_warning_level

    raw_gas = web3.eth.get_balance(hot_wallet.address)
    gas = raw_gas / (10**18)

    run_state.hot_wallet_gas = gas
    run_state.hot_wallet_address = hot_wallet.address

    if hot_wallet_gas_warning_level is not None:
        if gas < hot_wallet_gas_warning_level:
            # Set flag, issue warning (repeatably)
            gas_warn_message = f"Hot wallet running low on gas money.\nHot wallet address {hot_wallet.address}, gas is {gas} tokens, warning level is {hot_wallet_gas_warning_level} native tokens."
            logger.warning(gas_warn_message)
            run_state.hot_wallet_gas_warning_message = gas_warn_message
            return True
        elif gas >= hot_wallet_gas_warning_level:
            # Clear flag, issue warning ended
            if run_state.hot_wallet_gas_warning_message is not None:
                logger.warning(f"Hot wallet {hot_wallet.address} received top up - gas is now {gas}")
                run_state.hot_wallet_gas_warning_message = None

    return False



