import dataclasses
import datetime
from decimal import Decimal
from typing import Dict, List, Tuple

from eth_typing import HexAddress
from web3 import Web3

from smart_contracts_for_testing.portfolio import fetch_erc20_balances_decimal, DecimalisedHolding
from tradeexecutor.state.state import State, Portfolio, AssetIdentifier, ReservePosition
from tradingstrategy.pair import PairUniverse


ReserveMap = Dict[AssetIdentifier, Decimal]


@dataclasses.dataclass
class ReserveUpdateEvent:
    asset: AssetIdentifier
    tick: int
    updated_at: datetime.datetime
    past_balance: Decimal
    new_balance: Decimal


def update_wallet_balances(web3: Web3, address: HexAddress) -> Dict[HexAddress, DecimalisedHolding]:
    """Get raw balances of ERC-20 tokens.
    """
    return fetch_erc20_balances_decimal(web3, address)


def map_balances_to_trading_pairs(balances: Dict[HexAddress, Decimal], portfolio: Portfolio) -> Dict[HexAddress, Decimal]:
    """Get the balances for each """


def sync_reserves(
        web3: Web3,
        tick: int,
        clock: datetime.datetime,
        wallet_address: HexAddress,
        current_reserves: List[ReservePosition],
        supported_reserve_currencies: List[AssetIdentifier]) -> Tuple[ReserveMap, List[ReserveUpdateEvent]]:
    """Check the address for any incoming stablecoin transfers to see how much cash we have."""

    our_chain_id = web3.eth.chain_id

    # Get raw ERC-20 holdings of the address
    balances = update_wallet_balances(web3, wallet_address)

    reserves_per_token = {r.asset.address: r for r in current_reserves}

    events: ReserveUpdateEvent = []

    new_reserves: ReserveMap = {}

    for currency in supported_reserve_currencies:
        assert currency.chain_id == our_chain_id, f"Asset expects chain_id {currency.chain_id}, currently connected to {our_chain_id}"

        if currency.address in reserves_per_token:
            # We have an existing record of having this reserve
            current_value = reserves_per_token[currency.address].quantity
        else:
            current_value = Decimal(0)

        assert web3.isChecksumAddress(currency.address)
        decimal_holding = balances.get(currency.address)

        if (decimal_holding is not None) and (decimal_holding.value != current_value):
            evt = ReserveUpdateEvent(
                asset=currency,
                past_balance=current_value,
                new_balance=decimal_holding.value,
                tick=tick,
                updated_at=clock
            )
            events.append(evt)

            new_reserves[currency.address] = decimal_holding.value

    return new_reserves, events
