"""Test correct-accounts CLI command with real Aster CCXT API.

Following pattern from test_correct_accounts_derive.py.
"""

import datetime
import logging
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

ccxt = pytest.importorskip("ccxt", reason="ccxt package not installed")

from eth_defi.hotwallet import HotWallet
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ASTER_API_KEY") or not os.environ.get("ASTER_API_SECRET"),
    reason="Set ASTER_API_KEY and ASTER_API_SECRET for Aster CCXT tests",
)


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the CCXT exchange account strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/ccxt_exchange_account_strategy.py"


@pytest.fixture()
def strategy_file_anvil() -> Path:
    """Path to the minimal strategy module for Anvil chain CLI testing."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/minimal_ccxt_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    """Create a temporary state file path."""
    return Path(tempfile.mkdtemp()) / "test-correct-accounts-ccxt.json"


@pytest.fixture()
def environment(
    anvil,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
) -> dict:
    """Set up environment vars for correct-accounts CLI command.

    Following pattern from test_correct_accounts_derive.py.
    """
    import json
    ccxt_options = json.dumps({
        "apiKey": os.environ["ASTER_API_KEY"],
        "secret": os.environ["ASTER_API_SECRET"],
    })

    environment = {
        "EXECUTOR_ID": "test_correct_accounts_ccxt",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        # CCXT credentials
        "CCXT_EXCHANGE_ID": "aster",
        "CCXT_OPTIONS": ccxt_options,
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
    }
    return environment


@pytest.fixture()
def environment_anvil(
    anvil,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file_anvil: Path,
    usdc,
    dummy_token,
) -> dict:
    """Set up environment vars for init CLI command on Anvil chain.

    Uses the anvil strategy which is configured for ChainId.anvil (31337).
    """
    import json
    ccxt_options = json.dumps({
        "apiKey": os.environ["ASTER_API_KEY"],
        "secret": os.environ["ASTER_API_SECRET"],
    })

    environment = {
        "EXECUTOR_ID": "test_ccxt_cli_start",
        "STRATEGY_FILE": strategy_file_anvil.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        # CCXT credentials
        "CCXT_EXCHANGE_ID": "aster",
        "CCXT_OPTIONS": ccxt_options,
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
        # Token addresses for strategy to use
        "TEST_USDC_ADDRESS": usdc.address,
        "TEST_DUMMY_TOKEN_ADDRESS": dummy_token.address,
        # Run single cycle for start command tests
        "RUN_SINGLE_CYCLE": "true",
    }
    return environment


def _create_test_state_with_ccxt_position(state_file: Path) -> tuple[State, Decimal]:
    """Create test state with an Aster CCXT exchange account position.

    Creates a position with tracked value at 99% of actual Aster balance
    so that correct-accounts will detect a difference.

    :return: Tuple of (state, actual_value from Aster API)
    """
    from tradeexecutor.exchange_account.ccxt_exchange import (
        create_ccxt_exchange,
        create_ccxt_account_value_func,
    )
    from tradeexecutor.state.identifier import (
        AssetIdentifier,
        TradingPairIdentifier,
        TradingPairKind,
    )
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType

    # Set up CCXT exchange
    exchange = create_ccxt_exchange("aster", {
        "apiKey": os.environ["ASTER_API_KEY"],
        "secret": os.environ["ASTER_API_SECRET"],
    })

    # Create exchange account pair
    usdc = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    aster_account = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="ASTER-ACCOUNT",
        decimals=6,
    )
    exchange_account_pair = TradingPairIdentifier(
        base=aster_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="ccxt_aster",
        other_data={
            "exchange_protocol": "ccxt",
            "ccxt_account_id": "aster_main",
            "ccxt_exchange_id": "aster",
            "exchange_is_testnet": False,
        },
    )

    # Get actual account value
    exchanges = {"aster_main": exchange}
    account_value_func = create_ccxt_account_value_func(exchanges)
    actual_value = account_value_func(exchange_account_pair)
    assert actual_value > 0, "Aster account should have funds"

    # Create position with 99% of actual value to trigger sync
    tracked_value = (actual_value * Decimal("0.99")).quantize(Decimal("0.01"))

    state = State()
    opened_at = datetime.datetime(2024, 1, 1)

    position = TradingPosition(
        position_id=1,
        pair=exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=exchange_account_pair.quote,
    )

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=tracked_value,
        planned_price=1.0,
        planned_reserve=tracked_value,
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=tracked_value,
        executed_reserve=tracked_value,
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    state.portfolio.open_positions[1] = position

    # Save state file
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    return state, actual_value


def test_correct_accounts_ccxt(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
):
    """Test correct-accounts syncs Aster CCXT exchange account positions.

    1. Create state file with exchange account position (tracked value 99% of actual)
    2. Invoke correct-accounts CLI command
    3. Verify state file was updated with balance correction
    """
    initial_state, actual_value = _create_test_state_with_ccxt_position(state_file)
    initial_quantity = initial_state.portfolio.open_positions[1].get_quantity()

    logger.info("Initial tracked value: %s, actual Aster value: %s", initial_quantity, actual_value)

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])

        assert e.value.code == 0, f"CLI command failed with exit code {e.value.code}"

    # Verify state was updated
    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]
    final_quantity = final_position.get_quantity()

    logger.info("Final quantity: %s (expected ~%s)", final_quantity, actual_value)

    # Balance should have been corrected (within small tolerance for API timing)
    assert abs(float(final_quantity) - float(actual_value)) < float(actual_value) * 0.02, \
        f"Expected quantity ~{actual_value}, got {final_quantity}"

    # Balance update should be recorded
    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1

    logger.info("correct-accounts CCXT test passed")


def test_ccxt_cli_start(
    logger: logging.Logger,
    environment_anvil: dict,
    state_file: Path,
):
    """Test init CLI command with CCXT exchange account support.

    Following pattern from test_derive_cli_start.
    Tests that the strategy can be initialised with CCXT environment variables.
    """
    cli = get_command(app)

    with patch.dict(os.environ, environment_anvil, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    # Verify state was created
    state = State.read_json_file(state_file)

    assert state is not None
    assert state.sync is not None

    logger.info("test_ccxt_cli_start passed")
