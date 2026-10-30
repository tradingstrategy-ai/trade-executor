"""Flow-neutral Lighter account profit through the Typer CLI on an Ethereum fork.

The state unit tests drive the share-price code directly. This test runs the
same kinds of event through the commands an operator uses, on the fixed
Ethereum fork with the simulated Lighter Core: deposits and transfers with
``perform-test-trade``, withdrawals with ``lighter-move-funds``, gains and
losses injected into the fake sequencer and booked by ``correct-accounts`` or
``lagoon-settle``. The fake charges a 0.5 USDC fee per transfer between
accounts. A withdrawal fee is covered by the state unit tests only:
``lighter-move-funds`` refuses a withdrawal whose credit differs from the
request.

After every step each account's USD profit must equal the gains and losses
injected into it minus the fees it paid, and the share price of every account
the step did not revalue must stay put, except a fee payer's, which falls by
its fee over its value. At the end the accounts' profits add up to the
deployer's USDC change.
"""

import io
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_WHALE
from pytest import CaptureFixture, MonkeyPatch

from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.position_internal_share_price import calculate_flow_neutral_profit_usd
from tradeexecutor.testing import lighter_manual
from tradeexecutor.testing.lighter_tutorial import (
    SIMULATED_TRANSFER_FEE_USDC,
    STRATEGY_DIR,
    Tutorial,
    deploy,
    get_base_environment,
    get_cli_environment,
    load_tutorial_config,
    prepare_tutorial,
    run_cli,
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

#: Test capital.
SUBSCRIPTION_USDC = Decimal(20)
#: Fee per transfer between Lighter accounts in the fake sequencer.
FEE = SIMULATED_TRANSFER_FEE_USDC
#: Absolute tolerance of the profit and share price checks.
TOLERANCE = 1e-6


@dataclass(slots=True)
class Figures:
    """One account's share price and value, read from the saved state."""

    #: Internal share price as last updated, 0 before the account is tracked.
    share_price: float
    #: Booked account value.
    value: float


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
    """Prepare a two-sub-account vault in simulate mode on the shared fork."""
    monkeypatch.setenv("SIMULATE", "true")
    monkeypatch.setenv("LIGHTER_SIMULATE_RPC_URL", anvil_ethereum.json_rpc_url)
    monkeypatch.setenv("LIGHTER_RUN_DIR", str(tmp_path / "run"))
    config = load_tutorial_config(executor_id="lighter-profit", strategy_file=STRATEGY_DIR / "lighter_multi_subaccount_strategy.py", subscription_usdc=SUBSCRIPTION_USDC)
    tutorial = prepare_tutorial(config)
    try:
        yield tutorial
    finally:
        tutorial.close()


def manual(capsys: CaptureFixture, *args: str) -> str:
    """Run one ``lighter_manual`` subcommand in-process, so it sees the fake sequencer, and return its result line."""
    capsys.readouterr()
    code = lighter_manual.main(list(args))
    out = capsys.readouterr().out.strip()
    assert code == 0, f"lighter_manual {' '.join(args)} failed: {out}"
    return out


class Ledger:
    """Gains, losses and fees per account, and the checks run after every step.

    :param tutorial:
        The deployed simulated vault.
    """

    def __init__(self, tutorial: Tutorial):
        self.tutorial = tutorial
        master, trend, arb = tutorial.account_indices
        self.names = {master: "master", trend: "trend", arb: "arb"}
        self.gains = {name: Decimal(0) for name in self.names.values()}
        self.fees = {name: Decimal(0) for name in self.names.values()}
        self.figures = self.read_figures()

    def read_positions(self) -> dict[str, TradingPosition]:
        """Read each account's latest position from the saved state."""
        state = State.read_json_file(self.tutorial.state_file)
        positions = {}
        for position in state.portfolio.get_all_positions():
            if position.is_exchange_account():
                positions[self.names[int(position.pair.get_exchange_account_id())]] = position
        return positions

    def read_figures(self) -> dict[str, Figures]:
        """Read each account's share price and value from the saved state."""
        return {
            name: Figures(
                share_price=position.share_price_state.current_share_price if position.share_price_state else 0.0,
                value=float(position.get_quantity()),
            )
            for name, position in self.read_positions().items()
        }

    def credit(self, account: str, amount: Decimal) -> None:
        """Change an account's balance in the fake sequencer, a gain or a loss."""
        index = next(index for index, name in self.names.items() if name == account)
        self.tutorial.simulation.fake_core.get_account(index).collateral += amount
        self.gains[account] += amount

    def check(self, step: str, revalued: set[str] = frozenset(), transfer_fees: dict[str, Decimal] | None = None, deposited: dict[str, Decimal] | None = None) -> None:
        """Check every account after a step.

        :param revalued:
            Accounts whose share price the step may move: valued, or restarted after a total loss.
        :param transfer_fees:
            Transfer fees paid in the step, by account; their share price falls by the fee over the
            account's value when the fee was charged.
        :param deposited:
            Deposits made in the step before its transfer, by account, which the fee's value includes.
        """
        transfer_fees = transfer_fees or {}
        deposited = deposited or {}
        before, after = self.figures, self.read_figures()
        for name, position in self.read_positions().items():
            if position.share_price_state is None:
                # A sub-account is tracked from its first funding.
                assert self.gains[name] == 0 and self.fees[name] == 0, f"{step}: {name} has gains or fees but no share-price state"
                continue
            profit = calculate_flow_neutral_profit_usd(position)
            expected_profit = float(self.gains[name] - self.fees[name])
            assert profit is not None and profit == pytest.approx(expected_profit, abs=TOLERANCE), f"{step}: {name} profit {profit}, expected gains minus fees {expected_profit}"
            if name in revalued or before[name].share_price == 0:
                continue
            fee = float(transfer_fees.get(name, 0))
            value_at_fee = before[name].value + float(deposited.get(name, 0))
            expected_price = before[name].share_price * (value_at_fee - fee) / value_at_fee if fee else before[name].share_price
            assert after[name].share_price == pytest.approx(expected_price, abs=TOLERANCE), f"{step}: {name} share price moved from {before[name].share_price} to {after[name].share_price}, expected {expected_price}"
        self.figures = after


@pytest.mark.timeout(900)
def test_cli_lighter_account_profit_is_flow_neutral(tutorial: Tutorial, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
    """Each Lighter account's return and USD profit stay independent of the capital the CLI moves.

    The steps cover a deposit after a gain, a transfer between sub-accounts, a
    partial withdrawal after a loss, a total loss followed by re-funding, a
    full exit and re-entry, and a final withdrawal of every account, with gains booked
    by both ``correct-accounts`` and ``lagoon-settle``.

    1. Deploy, initialise, subscribe and top up the master, which the activation deposit seeds
    2. Fund trend from the Safe and book a gain on it with correct-accounts
    3. Fund arb, book a gain on it with lagoon-settle, then fund it again
    4. Move USDC from trend to arb with --lighter-transfer
    5. Book a loss on arb, then withdraw half of it
    6. Lose all of trend, then fund it again
    7. Withdraw all of arb, fund it again and book a gain on it
    8. Withdraw every account, redeem, and compare the summed profit with the deployer's USDC change
    """
    deployer_start = tutorial.usdc.fetch_balance_of(tutorial.deployer.address)
    for key, value in get_base_environment(tutorial).items():
        monkeypatch.setenv(key, value)

    # 1. Deploy, initialise, subscribe and top up the master, which the activation deposit seeds
    deploy(tutorial, "Lighter profit test", "LPRO")
    master, trend, arb = tutorial.account_indices
    signers = ",".join(str(tutorial.get_handover_file(name)) for name in ("trend", "arb"))
    env = get_cli_environment(tutorial, LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES=signers)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run_cli(["init"], env)
    run_cli(["correct-accounts", "--process-redemption"], env)
    marker = tutorial.config.run_dir / "deposit-requested"
    assert manual(capsys, "request-deposit", str(SUBSCRIPTION_USDC), "--marker", str(marker)) == "requested"
    run_cli(["lagoon-settle"], env)
    manual(capsys, "claim-shares")
    run_cli(["perform-test-trade", "--single-pair", "--buy-only", "--amount", "4"], env)
    ledger = Ledger(tutorial)
    ledger.check("subscription and master top-up")

    def fund(account: str, amount: str, step: str, revalued: set[str] = frozenset()) -> None:
        """Deposit to the master and fund a sub-account, the master paying the fee, then check."""
        run_cli(["perform-test-trade", "--pair", f"(ethereum, lighter, LIGHTER-{account}, USDC)", "--buy-only", "--amount", amount], env)
        ledger.fees["master"] += FEE
        ledger.check(step, revalued=revalued, transfer_fees={"master": FEE}, deposited={"master": Decimal(amount)})

    def withdraw(account: str, amount: Decimal) -> None:
        """Withdraw from an account to the Safe with lighter-move-funds."""
        options = [] if account == "master" else ["--lighter-account-index", str({"trend": trend, "arb": arb}[account])]
        monkeypatch.setattr("sys.stdin", io.StringIO(f"w\n{amount}\ny\n"))
        run_cli(["lighter-move-funds", *options], env)

    # 2. Fund trend from the Safe and book a gain on it with correct-accounts
    fund("trend", "4", "fund trend")
    ledger.credit("trend", Decimal("0.4"))
    run_cli(["correct-accounts"], env)
    ledger.check("gain on trend", revalued={"trend"})

    # 3. Fund arb, book a gain on it with lagoon-settle, then fund it again
    fund("arb", "3", "fund arb")
    ledger.credit("arb", Decimal("0.3"))
    run_cli(["lagoon-settle"], env)
    ledger.check("gain on arb booked by lagoon-settle", revalued={"arb"})
    fund("arb", "3.3", "fund arb after its gain")
    assert ledger.figures["arb"].share_price == pytest.approx(1.1), "A deposit after a gain mints at the higher price"

    # 4. Move USDC from trend to arb with --lighter-transfer
    run_cli(["perform-test-trade", "--lighter-transfer", "trend:arb", "--buy-only", "--amount", "2.2"], env)
    ledger.fees["trend"] += FEE
    ledger.check("transfer trend to arb", transfer_fees={"trend": FEE})

    # 5. Book a loss on arb, then withdraw half of it
    ledger.credit("arb", Decimal("-0.88"))
    run_cli(["correct-accounts"], env)
    ledger.check("loss on arb", revalued={"arb"})
    withdraw("arb", Decimal("3.96"))
    ledger.check("partial withdrawal from arb")

    # 6. Lose all of trend, then fund it again
    ledger.credit("trend", -ledger.tutorial.simulation.fake_core.get_account(trend).collateral)
    run_cli(["correct-accounts"], env)
    ledger.check("total loss on trend", revalued={"trend"})
    fund("trend", "2", "re-fund trend after its total loss", revalued={"trend"})
    assert ledger.figures["trend"].share_price == pytest.approx(1.0), "A deposit after a total loss starts a new series"

    # 7. Withdraw all of arb, fund it again and book a gain on it
    withdraw("arb", Decimal("3.96"))
    ledger.check("full withdrawal from arb")
    fund("arb", "2", "re-fund arb")
    ledger.credit("arb", Decimal("0.2"))
    run_cli(["correct-accounts"], env)
    ledger.check("gain on arb after re-entry", revalued={"arb"})

    # 8. Withdraw every account, redeem, and compare the summed profit with the deployer's USDC change
    for account in ("trend", "arb", "master"):
        withdraw(account, ledger.read_positions()[account].get_quantity())
        ledger.check(f"withdraw {account}")
    run_cli(["lagoon-redeem"], env)
    profits = {name: calculate_flow_neutral_profit_usd(position) for name, position in ledger.read_positions().items()}
    # Master: five transfer fees; trend: +0.4 - 1.7 and one fee; arb: +0.3 - 0.88 + 0.2.
    assert profits == pytest.approx({"master": -2.5, "trend": -1.8, "arb": -0.38}, abs=TOLERANCE)
    deployer_change = tutorial.usdc.fetch_balance_of(tutorial.deployer.address) - deployer_start
    assert float(deployer_change) == pytest.approx(sum(profits.values()), abs=1e-5)
