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
       ).call(block_identifier=401_729_535)

Note: PnL currently uses live GMX oracle prices. Block number records
which chain state was read. Will switch to per-block oracle when available.

Test accounts at block 401_729_535
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``0x1640e916e10610Ba39aAC5Cd8a08acF3cCae1A4c``
    9 USDC-collateralised positions (mixed long/short across ARB, LINK, SOL,
    DOGE, BTC, AAVE, PEPE, XRP markets), ~$978K USDC reserves, ~$272K total
    collateral.

``0x9dd1497FF0775bab1FAEb45ea270F66b11496dDf``
    1 ETH-collateralised short position (~588 ETH collateral, ~$2.7M notional),
    zero USDC/WETH wallet reserves.  Tests non-USDC collateral handling.
"""

import logging
import os
from decimal import Decimal

import pytest
from web3 import HTTPProvider, Web3

from eth_defi.chain import install_chain_middleware
from eth_defi.compat import native_datetime_utc_now
from eth_defi.gas import node_default_gas_price_strategy
from eth_defi.gmx.valuation import fetch_gmx_total_equity
from eth_defi.provider.anvil import fork_network_anvil
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

pytestmark = pytest.mark.skipif(
    not os.environ.get("JSON_RPC_ARBITRUM"),
    reason="JSON_RPC_ARBITRUM environment variable not set",
)

#: Arbitrum USDC (native) address
USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

#: Account with 9 USDC-collateralised GMX positions (mixed long/short)
ACCOUNT_USDC_POSITIONS = "0x1640e916e10610Ba39aAC5Cd8a08acF3cCae1A4c"

#: Account with 1 ETH-collateralised short position, no wallet reserves
ACCOUNT_ETH_SHORT = "0x9dd1497FF0775bab1FAEb45ea270F66b11496dDf"

#: Fixed fork block for deterministic tests
FORK_BLOCK = 401_729_535

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161


@pytest.fixture()
def anvil_arbitrum():
    """Launch an Anvil mainnet fork of Arbitrum at a fixed block."""
    rpc_url = os.environ["JSON_RPC_ARBITRUM"]
    launch = fork_network_anvil(
        rpc_url,
        fork_block_number=FORK_BLOCK,
        test_request_timeout=100,
        launch_wait_seconds=60,
    )
    try:
        yield launch.json_rpc_url
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3(anvil_arbitrum):
    """Web3 connected to the Anvil fork."""
    web3 = Web3(HTTPProvider(anvil_arbitrum, request_kwargs={"timeout": 100}))
    install_chain_middleware(web3)
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


def test_gmx_valuation_pipeline_usdc_positions(web3, usdc_asset):
    """Test the full valuation pipeline for a USDC-collateralised GMX account.

    Verifies:
    - GMX value func returns correct position equity via fetch_gmx_total_equity
    - ExchangeAccountValuator captures and persists block_number
    - position.valuation_updates is populated
    - BalanceUpdate events have block_number set
    """
    # 1. Get reference value directly from fetch_gmx_total_equity
    usdc_token = fetch_erc20_details(web3, USDC_ADDRESS)
    reference = fetch_gmx_total_equity(
        web3=web3,
        account=ACCOUNT_USDC_POSITIONS,
        reserve_tokens=[usdc_token],
        block_identifier=FORK_BLOCK,
    )
    # Reserves are deterministic at the fork block
    assert reference.reserves == pytest.approx(Decimal("978_163.293624"), rel=Decimal("0.001"))
    # Positions must be positive (collateral alone is ~$272K)
    assert reference.positions > Decimal("200_000")

    # 2. Create state with exchange account position
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

    # 3. Create GMX value func → pricing model → valuator (with web3)
    value_func = create_gmx_account_value_func(
        web3=web3,
        safe_address=ACCOUNT_USDC_POSITIONS,
    )
    pricing_model = ExchangeAccountPricingModel(value_func)
    valuator = ExchangeAccountValuator(pricing_model, web3=web3)

    # 4. Run valuation
    evt = valuator(ts, position)

    # 5. Verify ValuationUpdate has block_number and is appended
    assert evt.block_number is not None
    assert evt.block_number >= FORK_BLOCK
    assert len(position.valuation_updates) == 1
    assert position.valuation_updates[0] is evt

    # 6. Verify position value matches reference (positions only, no reserves)
    assert evt.new_value == pytest.approx(float(reference.positions), rel=0.05)

    # 7. Since we started with 0 quantity and got a non-zero value,
    #    there should be a BalanceUpdate with block_number
    assert len(position.balance_updates) > 0
    balance_evt = list(position.balance_updates.values())[-1]
    assert balance_evt.block_number is not None
    assert balance_evt.block_number >= FORK_BLOCK


def test_gmx_valuation_pipeline_eth_short(web3, usdc_asset):
    """Test the full valuation pipeline for an ETH-collateralised short position.

    This account has no USDC reserves but one large ETH short.
    Verifies non-USDC collateral handling through the pipeline.
    """
    # 1. Get reference value
    reference = fetch_gmx_total_equity(
        web3=web3,
        account=ACCOUNT_ETH_SHORT,
        reserve_tokens=[],
        block_identifier=FORK_BLOCK,
    )
    # Position value should be substantial (collateral ~$1.2M + short PnL)
    assert reference.positions > Decimal("1_500_000")
    assert reference.reserves == Decimal(0)

    # 2. Create state with exchange account position
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

    # 3. Create GMX value func → pricing model → valuator
    value_func = create_gmx_account_value_func(
        web3=web3,
        safe_address=ACCOUNT_ETH_SHORT,
    )
    pricing_model = ExchangeAccountPricingModel(value_func)
    valuator = ExchangeAccountValuator(pricing_model, web3=web3)

    # 4. Run valuation
    evt = valuator(ts, position)

    # 5. Verify block tracking
    assert evt.block_number is not None
    assert evt.block_number >= FORK_BLOCK
    assert len(position.valuation_updates) == 1

    # 6. Verify value matches reference (positions only)
    assert evt.new_value == pytest.approx(float(reference.positions), rel=0.05)

    # 7. Verify BalanceUpdate block tracking
    assert len(position.balance_updates) > 0
    balance_evt = list(position.balance_updates.values())[-1]
    assert balance_evt.block_number is not None
