"""Black-box ``perform-test-trade`` for a Lagoon vault with Lighter sub-accounts.

An operator verifies a fresh deployment with ``perform-test-trade``. For a
Lighter vault the test trade is a custody round trip: Safe to the master
account and back, and for a sub-account through the master. The vault runs
on the fixed Ethereum fork through the tutorial helpers; only Lighter's
sequencer is faked.
"""

import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE
from pytest import MonkeyPatch

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.testing.lighter_tutorial import (
    SIMULATED_TRANSFER_FEE_USDC,
    STRATEGY_DIR,
    Tutorial,
    deploy,
    get_cli_environment,
    initialise_state,
    load_tutorial_config,
    prepare_tutorial,
    run_cli,
    subscribe,
)

#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: API key required by the strategy factory even though this strategy has no market data.
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM or not TRADING_STRATEGY_API_KEY,
        reason="JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY are required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]

#: Subscription kept in the Safe for the test trades.
SUBSCRIPTION_USDC = Decimal(9)
#: First test trade, left on the master as headroom for the transfer fees.
HEADROOM_USDC = Decimal(2)


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the test."""
    launch = anvil_fork_pool.get_launch(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
        unlocked_addresses=[USDC_WHALE[1]],
        inherit_stdio=True,
    )
    snapshot = evm_snapshot_revert(launch)
    next(snapshot)
    try:
        yield launch
    finally:
        next(snapshot, None)


@pytest.fixture()
def tutorial(anvil_ethereum: AnvilLaunch, tmp_path: Path, monkeypatch: MonkeyPatch) -> Tutorial:
    """Prepare the transfer tutorial's vault in simulate mode on the shared fork."""
    monkeypatch.setenv("SIMULATE", "true")
    monkeypatch.setenv("LIGHTER_SIMULATE_RPC_URL", anvil_ethereum.json_rpc_url)
    monkeypatch.setenv("LIGHTER_RUN_DIR", str(tmp_path / "run"))
    config = load_tutorial_config(executor_id="lighter-test-trade", strategy_file=STRATEGY_DIR / "lighter_scripted_transfers_strategy.py", subscription_usdc=SUBSCRIPTION_USDC)
    tutorial = prepare_tutorial(config)
    try:
        yield tutorial
    finally:
        tutorial.close()


@pytest.mark.timeout(900)
def test_cli_perform_test_trade_lighter_sub_accounts(tutorial: Tutorial) -> None:
    """Run perform-test-trade against the master and every sub-account of a Lighter vault.

    The test trade must use the same custody trades ``start`` makes, so a
    deployment that passes it can run the strategy, and it must refuse to
    run without the operator record instead of half-moving funds.

    1. Deploy a vault with two sub-accounts and initialise state.
    2. Subscribe and settle without running the strategy, so the Safe keeps the subscription.
    3. Deposit to the master with --single-pair --buy-only, leaving headroom for the transfer fees.
    4. Round-trip every account with --all-pairs and check the trades, the fees and check-accounts.
    5. Check that the command refuses to run without the operator record and creates no trade.
    """
    # 1. Deploy a vault with two sub-accounts and initialise state.
    deploy(tutorial, "Lighter test trade", "LTT")
    initialise_state(tutorial)
    master, trend, arb = tutorial.account_indices
    # The fake sequencer stands in for Lighter's L2, which Anvil cannot run; its balances are the "real" ones here.
    fake = tutorial.simulation.fake_core
    activation = fake.get_account(master).collateral

    # 2. Subscribe and settle without running the strategy, so the Safe keeps the subscription.
    subscribe(tutorial, settle_with_start=False)
    assert tutorial.usdc.fetch_balance_of(tutorial.safe_address) == SUBSCRIPTION_USDC

    # 3. Deposit to the master with --single-pair --buy-only, leaving headroom for the transfer fees.
    run_cli(["perform-test-trade", "--single-pair", "--buy-only", "--amount", str(HEADROOM_USDC)], get_cli_environment(tutorial))
    state = State.read_json_file(tutorial.state_file)
    [deposit] = [trade for trade in state.portfolio.get_all_trades() if trade.is_exchange_account_transfer()]
    assert deposit.is_success() and deposit.is_buy() and deposit.notes.startswith("perform-test-trade")
    assert TradeFlag.test_trade not in deposit.flags, "The vault's permanent master position must not become a test position"
    assert fake.get_account(master).collateral == activation + HEADROOM_USDC
    safe_before_round_trips = tutorial.usdc.fetch_balance_of(tutorial.safe_address)
    assert safe_before_round_trips == SUBSCRIPTION_USDC - HEADROOM_USDC

    # 4. Round-trip every account with --all-pairs and check the trades, the fees and check-accounts.
    run_cli(["perform-test-trade", "--all-pairs", "--amount", "1"], get_cli_environment(tutorial))
    state = State.read_json_file(tutorial.state_file)
    trades_by_account = {}
    for trade in state.portfolio.get_all_trades():
        if trade.is_exchange_account_transfer() and trade.trade_id != deposit.trade_id:
            trades_by_account.setdefault(int(trade.pair.get_exchange_account_id()), []).append(trade)
    assert all(trade.is_success() for trades in trades_by_account.values() for trade in trades)
    # The master gets every deposit and its own withdrawal; each sub-account gets a funding and a withdrawal.
    assert [trade.is_buy() for trade in trades_by_account[master]] == [True, False, True, True]
    for index in (trend, arb):
        assert [(trade.is_internal_account_transfer(), trade.is_buy()) for trade in trades_by_account[index]] == [(True, True), (False, False)]
        assert fake.get_account(index).collateral == 0
    assert tutorial.usdc.fetch_balance_of(tutorial.safe_address) == safe_before_round_trips
    assert fake.get_account(master).collateral == activation + HEADROOM_USDC - 2 * SIMULATED_TRANSFER_FEE_USDC
    master_position = state.portfolio.get_position_by_id(deposit.position_id)
    assert master_position.get_quantity() == fake.get_account(master).collateral
    run_cli(["check-accounts"], get_cli_environment(tutorial))

    # 5. Check that the command refuses to run without the operator record and creates no trade.
    trade_count = len(list(state.portfolio.get_all_trades()))
    environment = get_cli_environment(tutorial)
    del environment["LIGHTER_OPERATOR_RECORD_FILE"]
    with pytest.raises(ValueError, match="LIGHTER_OPERATOR_RECORD_FILE"):
        run_cli(["perform-test-trade", "--single-pair"], environment)
    assert len(list(State.read_json_file(tutorial.state_file).portfolio.get_all_trades())) == trade_count
