"""Integration tests for GMX account valuation via fetch_gmx_total_equity().

Tests the full valuation pipeline on an Anvil mainnet fork at a fixed block,
using real GMX positions to verify that:

- ``create_gmx_account_value_func()`` returns correct position equity
- ``ExchangeAccountValuator`` captures block numbers in state
- ``ValuationUpdate`` and ``BalanceUpdate`` events have ``block_number`` set
- ``position.valuation_updates`` is populated (regression for missing append)

Requires ``JSON_RPC_ARBITRUM`` environment variable pointing to an archive node.

Manual cross-validation
-----------------------

Position data (collateral, size, entry price) is read on-chain at the fork
block and is deterministic.  PnL uses *live* GMX oracle prices, so position
values will shift between test runs.  To manually cross-validate:

1. Open https://app.gmx.io/#/actions/<account> for the test accounts
   listed below.

2. Use the GMX REST API v2 to fetch live positions::

       from eth_defi.gmx.api import GMXAPI
       api = GMXAPI(chain="arbitrum")
       positions = api.get_positions("0x1640e916e10610Ba39aAC5Cd8a08acF3cCae1A4c")

3. Verify on-chain position data at the fork block::

       from eth_defi.gmx.contracts import get_reader_contract, get_contract_addresses
       reader = get_reader_contract(web3, "arbitrum")
       addresses = get_contract_addresses("arbitrum")
       positions = reader.functions.getAccountPositions(
           addresses.datastore, account, 0, 100
       ).call(block_identifier=484_000_000)

Note: PnL currently uses live GMX oracle prices. Block number records
which chain state was read. Will switch to per-block oracle when available.

Test accounts at block 484_000_000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``0x1640e916e10610Ba39aAC5Cd8a08acF3cCae1A4c``
    USDC-collateralised positions with ~$1.267M USDC reserves and a positive
    net GMX position value through the v2.2c Reader.

``0x9dd1497FF0775bab1FAEb45ea270F66b11496dDf``
    ETH-collateralised short position with a >$2M net GMX position value.
    Tests non-USDC collateral handling.
"""

import os
from decimal import Decimal

import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.gas import node_default_gas_price_strategy
from eth_defi.gmx.contracts import get_reader_contract
from eth_defi.gmx.valuation import fetch_gmx_total_equity
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.token import fetch_erc20_details

from tradeexecutor.exchange_account.gmx import (
    create_gmx_account_value_func,
    create_gmx_exchange_account_pair,
)
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("JSON_RPC_ARBITRUM"),
        reason="JSON_RPC_ARBITRUM environment variable not set",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:arbitrum:484000000"),
]

#: Arbitrum USDC (native) address
USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

#: Account with 9 USDC-collateralised GMX positions (mixed long/short)
ACCOUNT_USDC_POSITIONS = "0x1640e916e10610Ba39aAC5Cd8a08acF3cCae1A4c"

#: Account with an ETH-collateralised GMX position
ACCOUNT_ETH_SHORT = "0x9dd1497FF0775bab1FAEb45ea270F66b11496dDf"

#: Fixed fork block after the v2.2c Reader deployment at 483,924,493.
FORK_BLOCK = 484_000_000

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161

#: v2.2c Reader, deployed before :py:data:`FORK_BLOCK` and matched by the
#: vendored v2.2c ``Reader.json`` ABI.
GMX_READER_AT_FORK_BLOCK = "0xfA26cBb46e2614609406de08CA1Dc7f70a684184"


@pytest.fixture()
def web3(anvil_fork_pool: AnvilForkPool) -> Web3:
    """Create a Web3 client on the shared fixed-block v2.2c GMX fork."""
    web3 = anvil_fork_pool.get_web3(
        os.environ["JSON_RPC_ARBITRUM"],
        FORK_BLOCK,
        web3_retries=1,
        web3_http_timeout=(3, 100.0),
        test_request_timeout=100,
        launch_wait_seconds=60,
    )
    assert get_reader_contract(web3, "arbitrum").address == GMX_READER_AT_FORK_BLOCK
    web3.eth.set_gas_price_strategy(node_default_gas_price_strategy)
    return web3


@pytest.fixture()
def usdc_asset() -> AssetIdentifier:
    """USDC as an AssetIdentifier for state operations."""
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


def test_gmx_valuation_pipeline_usdc_positions(web3: Web3, usdc_asset: AssetIdentifier):
    """Test the full valuation pipeline for a USDC-collateralised GMX account.

    1. Obtain a direct v2.2c Reader reference value at the fixed fork block.
    2. Construct an exchange-account position and its pricing model.
    3. Value the position and verify the recorded value and block metadata.
    """
    # 1. Get reference value directly from fetch_gmx_total_equity
    usdc_token = fetch_erc20_details(web3, USDC_ADDRESS)
    reference = fetch_gmx_total_equity(
        web3=web3,
        account=ACCOUNT_USDC_POSITIONS,
        reserve_tokens=[usdc_token],
        block_identifier=FORK_BLOCK,
    )
    # GMX prices reserves using live signed oracle prices, so keep the
    # characterisation stable while still proving the account has USDC reserves.
    assert reference.reserves > Decimal("1_000_000")
    # The account has substantial positive v2.2c Reader position value.
    assert reference.positions > Decimal("150_000")

    # 2. Create the exchange-account state and pricing model.
    state = State()
    pair = create_gmx_exchange_account_pair(quote=usdc_asset)
    ts = native_datetime_utc_now()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=ts,
        pair=pair,
        reserve_currency=usdc_asset,
        reserve_amount=Decimal(0),
    )

    position = list(state.portfolio.open_positions.values())[0]
    assert position.is_exchange_account()
    assert len(position.valuation_updates) == 0

    value_func = create_gmx_account_value_func(
        web3=web3,
        safe_address=ACCOUNT_USDC_POSITIONS,
    )
    pricing_model = ExchangeAccountPricingModel(value_func)
    valuator = ExchangeAccountValuator(pricing_model, web3=web3)

    # 3. Run valuation and verify its value and block metadata.
    evt = valuator(ts, position)

    assert evt.block_number is not None
    assert evt.block_number >= FORK_BLOCK
    assert len(position.valuation_updates) == 1
    assert position.valuation_updates[0] is evt

    assert evt.new_value == pytest.approx(float(reference.positions), rel=0.05)

    assert len(position.balance_updates) > 0
    balance_evt = list(position.balance_updates.values())[-1]
    assert balance_evt.block_number is not None
    assert balance_evt.block_number >= FORK_BLOCK


def test_gmx_valuation_pipeline_eth_short(web3: Web3, usdc_asset: AssetIdentifier):
    """Test the full valuation pipeline for an ETH-collateralised short position.

    1. Obtain a direct v2.2c Reader reference for the ETH-collateralised account.
    2. Construct an exchange-account position and its pricing model.
    3. Value the position and verify the recorded value and block metadata.
    """
    # 1. Get reference value
    reference = fetch_gmx_total_equity(
        web3=web3,
        account=ACCOUNT_ETH_SHORT,
        reserve_tokens=[],
        block_identifier=FORK_BLOCK,
    )
    # Position value should be substantial for the ETH-collateralised short.
    assert reference.positions > Decimal("2_000_000")
    assert reference.reserves > Decimal(1)

    # 2. Create the exchange-account state and pricing model.
    state = State()
    pair = create_gmx_exchange_account_pair(quote=usdc_asset)
    ts = native_datetime_utc_now()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=ts,
        pair=pair,
        reserve_currency=usdc_asset,
        reserve_amount=Decimal(0),
    )

    position = list(state.portfolio.open_positions.values())[0]

    value_func = create_gmx_account_value_func(
        web3=web3,
        safe_address=ACCOUNT_ETH_SHORT,
    )
    pricing_model = ExchangeAccountPricingModel(value_func)
    valuator = ExchangeAccountValuator(pricing_model, web3=web3)

    # 3. Run valuation and verify its value and block metadata.
    evt = valuator(ts, position)

    assert evt.block_number is not None
    assert evt.block_number >= FORK_BLOCK
    assert len(position.valuation_updates) == 1

    assert evt.new_value == pytest.approx(float(reference.positions), rel=0.05)

    assert len(position.balance_updates) > 0
    balance_evt = list(position.balance_updates.values())[-1]
    assert balance_evt.block_number is not None
