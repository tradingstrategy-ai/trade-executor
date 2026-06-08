"""CLI integration tests for Ostium V1.5 async vault with trade-executor commands.

Tests that the following CLI commands handle async vault trades correctly:
- ``perform-test-trade``: Opens and closes a vault position with forced settlement on Anvil
- ``start`` with MAX_CYCLES: Runs decide_trades() which opens/closes vault positions
- ``check-accounts``: Reports no mismatches during pending settlement
- ``repair``: Does not touch vault_settlement_pending trades

These tests use the ``arbitrum-ostium-v15.py`` strategy module and verify
the full CLI pipeline works end-to-end with async vault deposits/redeems.
"""

import logging
import os
from pathlib import Path

import flaky
import pytest
from typer.main import get_command
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details, USDC_NATIVE_TOKEN, USDC_WHALE
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.utils.hex import hexbytes_to_hex_str


JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not TRADING_STRATEGY_API_KEY,
    reason="Set JSON_RPC_ARBITRUM and TRADING_STRATEGY_API_KEY to run",
)

FORK_BLOCK = 470_000_000


@pytest.fixture()
def anvil_arbitrum_fork() -> AnvilLaunch:
    """Arbitrum fork with unlocked USDC whale."""
    usdc_whale = USDC_WHALE[42161]
    launch = fork_network_anvil(
        JSON_RPC_ARBITRUM,
        fork_block_number=FORK_BLOCK,
        unlocked_addresses=[usdc_whale],
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3(anvil_arbitrum_fork) -> Web3:
    return create_multi_provider_web3(
        anvil_arbitrum_fork.json_rpc_url,
        default_http_timeout=(3, 250.0),
        retries=1,
    )


@pytest.fixture()
def hot_wallet(web3) -> HotWallet:
    """A test wallet funded with ETH and USDC."""
    hw = HotWallet.create_for_testing(web3, test_account_n=1, eth_amount=10)
    hw.sync_nonce(web3)

    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    tx_hash = usdc.contract.functions.transfer(
        hw.address, 500 * 10**6,
    ).transact({"from": USDC_WHALE[42161], "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return hw


@pytest.fixture()
def strategy_file() -> Path:
    """The Ostium V1.5 test strategy module."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "arbitrum-ostium-v15.py"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    return tmp_path / "ostium-v15-cli-test.json"


@pytest.fixture()
def environment(
    anvil_arbitrum_fork,
    hot_wallet,
    state_file,
    strategy_file,
    persistent_test_client,
) -> dict:
    """CLI environment for Ostium V1.5 hot wallet tests."""
    cache_path = persistent_test_client.transport.cache_path
    return {
        "EXECUTOR_ID": "test_vault_async_cli",
        "NAME": "test_vault_async_cli",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ARBITRUM": anvil_arbitrum_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),
        "MIN_GAS_BALANCE": "0.01",
        "GAS_BALANCE_WARNING_LEVEL": "0.0",
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.private_key),
        "CACHE_PATH": cache_path,
    }


@pytest.mark.skip(reason="perform-test-trade with forced settlement requires unlocked Ostium keeper on Anvil")
def test_ostium_v15_perform_test_trade(
    environment: dict,
    mocker,
    state_file: Path,
):
    """Perform-test-trade CLI opens/closes an Ostium V1.5 position on Anvil.

    1. Init strategy
    2. Run perform-test-trade (forces settlement on Anvil automatically)
    3. Verify state file has successful buy+sell trades
    """
    # 1. Init
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)
    cli.main(args=["init"], standalone_mode=False)

    # 2. Perform test trade (--all-vaults tests all vault pairs)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["perform-test-trade", "--all-vaults"], standalone_mode=False)

    # 3. Verify state
    state = State.read_json_file(state_file)
    trades = list(state.portfolio.get_all_trades())

    assert len(trades) >= 1
    buy_trade = trades[0]
    assert buy_trade.is_success(), f"Buy trade failed: {buy_trade.get_revert_reason()}"
    assert buy_trade.executed_quantity > 0

    if len(trades) >= 2:
        sell_trade = trades[1]
        assert sell_trade.is_success(), f"Sell trade failed: {sell_trade.get_revert_reason()}"


@flaky.flaky
def test_ostium_v15_start_one_cycle(
    environment: dict,
    mocker,
    state_file: Path,
):
    """Run one strategy cycle that deposits into Ostium V1.5 vault.

    1. Init strategy
    2. Run start with MAX_CYCLES=1
    3. Verify the buy trade was executed and is either success or vault_settlement_pending
       (on Anvil, settlement retry should resolve it within the same tick)
    """
    # 1. Init
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)
    cli.main(args=["init"], standalone_mode=False)

    # 2. Start with 1 cycle
    env_cycle = {**environment, "MAX_CYCLES": "1"}
    mocker.patch.dict("os.environ", env_cycle, clear=True)
    cli.main(args=["start"], standalone_mode=False)

    # 3. Verify the async vault deposit was executed
    state = State.read_json_file(state_file)
    trades = list(state.portfolio.get_all_trades())

    assert len(trades) >= 1
    buy_trade = trades[0]
    # Async vault flow: the trade calls requestDeposit() which is confirmed on-chain,
    # but settlement hasn't happened yet. The trade stays in vault_settlement_pending
    # until the next tick's settlement retry resolves it (after external settlement).
    # On Anvil with forced settlement, it might resolve within the same tick cycle.
    assert buy_trade.get_status() in (TradeStatus.success, TradeStatus.vault_settlement_pending), \
        f"Unexpected trade status: {buy_trade.get_status()}"


@flaky.flaky
def test_ostium_v15_check_accounts_cli(
    environment: dict,
    mocker,
    state_file: Path,
):
    """Run check-accounts after a vault deposit — should report no mismatches.

    1. Init + start (1 cycle to create a pending vault trade)
    2. Run check-accounts CLI
    3. Verify exit code 0 (no mismatches)
    """
    # 1. Init + one cycle
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)
    cli.main(args=["init"], standalone_mode=False)

    env_cycle = {**environment, "MAX_CYCLES": "1"}
    mocker.patch.dict("os.environ", env_cycle, clear=True)
    cli.main(args=["start"], standalone_mode=False)

    # 2. Check accounts — should exit 0 (no mismatches)
    # When a vault deposit is pending: on-chain USDC is lower (sent to vault via requestDeposit),
    # and state reserve is also lower (debited at trade start). So on-chain == expected.
    # Position has quantity 0 (no vault shares yet), on-chain also 0. No mismatch.
    env_check = {**environment, "SKIP_SAVE": "true"}
    mocker.patch.dict("os.environ", env_check, clear=True)
    try:
        cli.main(args=["check-accounts"], standalone_mode=False)
    except SystemExit as e:
        assert e.code == 0, f"check-accounts exited with code {e.code} — accounting mismatch detected"


@flaky.flaky
def test_ostium_v15_repair_cli(
    environment: dict,
    mocker,
    state_file: Path,
):
    """Run repair after a vault deposit — should not touch pending trades.

    1. Init + start (1 cycle)
    2. Run repair CLI
    3. Verify the pending trade was NOT repaired (still vault_settlement_pending or success)
    """
    # 1. Init + one cycle
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)
    cli.main(args=["init"], standalone_mode=False)

    env_cycle = {**environment, "MAX_CYCLES": "1"}
    mocker.patch.dict("os.environ", env_cycle, clear=True)
    cli.main(args=["start"], standalone_mode=False)

    # 2. Run repair (AUTO_APPROVE to skip y/n prompt)
    # The repair command has two paths that could touch our pending vault position:
    # - find_trades_to_be_repaired(): only selects is_failed() trades → skips pending
    # - repair_zero_quantity(): position has quantity 0 but we explicitly skip
    #   positions with vault_settlement_pending trades (fix in repair.py)
    env_repair = {**environment, "SKIP_SAVE": "true", "AUTO_APPROVE": "true"}
    mocker.patch.dict("os.environ", env_repair, clear=True)
    cli.main(args=["repair"], standalone_mode=False)

    # 3. Verify the vault_settlement_pending trade was NOT repaired or closed
    state = State.read_json_file(state_file)
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) >= 1
    buy_trade = trades[0]
    # Trade should remain in its original state — repair must not touch it
    assert buy_trade.get_status() in (TradeStatus.success, TradeStatus.vault_settlement_pending), \
        f"Repair incorrectly modified trade status to: {buy_trade.get_status()}"
