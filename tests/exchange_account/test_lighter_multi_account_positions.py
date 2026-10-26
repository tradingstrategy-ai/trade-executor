"""One trading position per Lighter account.

Pair equality and hashing compare base address, quote address and fee only,
so Lighter sub-account pairs need their own synthetic base asset. Otherwise a
second account's trades attach to the first account's position.
"""

from decimal import Decimal

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.exchange_account.lighter import (
    LIGHTER_ROLE_SUB_ACCOUNT,
    create_lighter_exchange_account_pair,
    create_lighter_exchange_account_pairs,
    get_lighter_master_pair,
    is_lighter_sub_account_pair,
    validate_lighter_exchange_account_pairs,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State

#: Master account index.
MASTER_INDEX = 700
#: Sub-account indices in creation order.
SUB_INDICES = (2**48 - 1000, 2**48 - 1001)


def _usdc() -> AssetIdentifier:
    """Return native Ethereum USDC."""
    return AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)


def test_lighter_accounts_are_separate_positions() -> None:
    """Every Lighter account gets a distinct pair and its own open position.

    The master keeps the identity used before sub-accounts existed, so
    running vaults need no state migration.

    1. Create the master and two sub-account pairs.
    2. Check the pairs are distinct by equality, hash, identifier, internal id and ticker, and the master is unchanged.
    3. Open one position per pair and check every portfolio lookup returns the right position.
    4. Reject universes with two masters or a sub-account of another master.
    """
    usdc = _usdc()

    # 1. Create the master and two sub-account pairs.
    master, sub_1, sub_2 = create_lighter_exchange_account_pairs(usdc, MASTER_INDEX, SUB_INDICES, strategy_labels={SUB_INDICES[0]: "trend"})

    # 2. Check the pairs are distinct by equality, hash, identifier, internal id and ticker, and the master is unchanged.
    assert master == create_lighter_exchange_account_pair(usdc, MASTER_INDEX)
    assert master.base.address == LIGHTER_L1_CONTRACT.lower() and master.internal_id == 1
    assert master != sub_1 and sub_1 != sub_2
    assert len({hash(master), hash(sub_1), hash(sub_2)}) == 3
    assert len({master.get_identifier(), sub_1.get_identifier(), sub_2.get_identifier()}) == 3
    assert len({master.internal_id, sub_1.internal_id, sub_2.internal_id}) == 3
    assert [pair.get_ticker() for pair in (master, sub_1, sub_2)] == ["LIGHTER-ACCOUNT-USDC", "LIGHTER-trend-USDC", f"LIGHTER-{SUB_INDICES[1]}-USDC"]
    assert sub_1.other_data["lighter_sub_account_id"] == "trend"
    assert [is_lighter_sub_account_pair(pair) for pair in (master, sub_1, sub_2)] == [False, True, True]
    assert validate_lighter_exchange_account_pairs([master, sub_1, sub_2]) == [master, sub_1, sub_2]
    assert get_lighter_master_pair([sub_2, master, sub_1]) == master

    # 3. Open one position per pair and check every portfolio lookup returns the right position.
    state = State()
    state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    for pair in (master, sub_1, sub_2):
        open_exchange_account_position(state=state, strategy_cycle_at=native_datetime_utc_now(), pair=pair, reserve_currency=usdc, reserve_amount=Decimal(0))
    positions = list(state.portfolio.open_positions.values())
    assert len(positions) == 3
    for pair, position in zip((master, sub_1, sub_2), positions):
        assert state.portfolio.get_open_position_for_pair(pair) is position
        assert state.portfolio.get_position_by_trading_pair(pair) is position
        assert position.pair.get_exchange_account_id() == pair.get_exchange_account_id()
    assert len(list(state.portfolio.get_all_traded_pairs())) == 3

    # 4. Reject universes with two masters or a sub-account of another master.
    with pytest.raises(ValueError, match="exactly one master"):
        validate_lighter_exchange_account_pairs([master, create_lighter_exchange_account_pair(usdc, MASTER_INDEX + 1)])
    foreign_sub = create_lighter_exchange_account_pair(usdc, SUB_INDICES[0] - 5, role=LIGHTER_ROLE_SUB_ACCOUNT, master_account_index=MASTER_INDEX + 1)
    with pytest.raises(ValueError, match="belong to the strategy's master"):
        validate_lighter_exchange_account_pairs([master, foreign_sub])
