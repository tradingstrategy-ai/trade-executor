"""Tutorial: book every Lighter custody transfer as a trade between positions.

A Lagoon vault's Safe owns a Lighter master account and two sub-accounts.
Each account is a separate trade-executor position, and every movement of
USDC is its own ``TradeExecution``: a deposit from the Safe to the master, a
transfer from one account to another, and a withdrawal from an account to
the Safe. This tutorial runs a scripted strategy whose ``decide_trades()``
makes all of those movements in a fixed order, winds the vault up, returns
every USDC to the deployer and then checks the executor state: one
successful trade per movement, on the right positions, with the right
amounts, fees and balance updates.

1. Deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, each with its own API key
2. Initialise executor state with one position per Lighter account
3. Subscribe to the vault
4. Run ``start`` cycles with ``lighter_scripted_transfers_strategy.py``:
   deposit the Safe cash to the master, fund both sub-accounts, move capital
   from sub-account 1 to sub-account 2 and from sub-account 2 to the master,
   then withdraw everything to the Safe
5. Redeem all shares with ``lagoon-redeem``
6. Sweep the Safe back to the deployer and reconcile the deployer's balances
7. Verify every scripted transfer in the executor state

The transfers sent from a sub-account are signed with that sub-account's
key, which this tutorial passes to ``start`` from the hand-over files
deployment wrote. In production those keys belong to the external traders;
giving them to the executor is optional.

Modes
-----

``SIMULATE=true`` forks Ethereum mainnet with Anvil at a fixed block and
replaces Lighter's sequencer with the fake used by the end-to-end tests.
Every Safe, guard, Lagoon and ZkLighter transaction still runs, so the
accounting is exercised end to end in a few minutes with no real money. The
fake charges a transfer fee, so the fee accounting is checked too. A
simulated run always deploys a fresh vault.

Without ``SIMULATE`` the tutorial runs on Ethereum mainnet and Lighter, like
``manual-trade-executor-lighter-subaccounts.py``. It needs a deployer with
5 USDC plus gas at the default unit, and each withdrawal takes 20 to 30
minutes. It resumes from ``progress.json`` in its run directory.

Example
-------

.. code-block:: shell

    source .local-test.env
    export JSON_RPC_ETHEREUM="https://..."

    # Dry run on an Anvil fork with the fake Lighter sequencer
    SIMULATE=true poetry run python scripts/lagoon/manual-trade-executor-lighter-transfers.py

    # Live run on Ethereum mainnet
    export LIGHTER_TEST_PRIVATE_KEY="0x..."
    poetry run python scripts/lagoon/manual-trade-executor-lighter-transfers.py

Optional environment variables
------------------------------

``LIGHTER_SCRIPTED_UNIT_USDC``
    Base amount of the fixed movements; the subscription is five units minus
    the 1 USDC activation deposit. Defaults to 1 live and 8 simulated.

``LIGHTER_SIMULATED_TRANSFER_FEE_USDC``
    Fee the fake sequencer charges per transfer. Defaults to 0.5.

``LIGHTER_RUN_DIR``
    Private working directory. Defaults below ``~/.tradingstrategy/examples``,
    or the temporary directory when simulating.

``LIGHTER_API_KEY_INDEX``, ``LIGHTER_WITHDRAW_TIMEOUT``, ``LIGHTER_TRANSFER_TIMEOUT``
    As in the sub-account tutorial.
"""

import logging
import os
from decimal import Decimal

from tabulate import tabulate

from tradeexecutor.state.state import State
from tradeexecutor.testing.lighter_scripted_transfers import build_lighter_transfer_schedule, get_scripted_progress, verify_lighter_scripted_transfers
from tradeexecutor.testing.lighter_tutorial import (
    STRATEGY_DIR,
    Tutorial,
    configure_logging,
    deploy,
    get_cli_environment,
    initialise_state,
    is_simulate_requested,
    load_tutorial_config,
    prepare_tutorial,
    redeem_all_shares,
    refund,
    run_cli,
    run_phases,
    run_start,
    subscribe,
)

logger = logging.getLogger(__name__)

#: Executor id and artefact prefix of this tutorial.
EXECUTOR_ID = "lighter-transfers"

#: Number of sub-accounts the schedule uses.
SUB_ACCOUNT_COUNT = 2

#: Schedule steps, each one ``start`` cycle.
SCHEDULE_STEPS = 5


def get_unit() -> Decimal:
    """Return the base amount of the scripted movements."""
    return Decimal(os.environ.get("LIGHTER_SCRIPTED_UNIT_USDC", "8" if is_simulate_requested() else "1"))


def get_strategy_environment(tutorial: Tutorial) -> dict[str, str]:
    """Return the ``start`` environment: the scripted unit and every account's signer."""
    return {
        "LIGHTER_SCRIPTED_UNIT_USDC": str(get_unit()),
        "LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES": ",".join(str(tutorial.get_handover_file(index)) for index in tutorial.account_indices[1:]),
    }


def run_scripted_cycles(tutorial: Tutorial) -> None:
    """Run ``start`` until the strategy has completed every schedule step."""
    for cycle in range(1, SCHEDULE_STEPS + 2):
        completed, problem = get_scripted_progress(State.read_json_file(tutorial.state_file), SCHEDULE_STEPS)
        if problem is not None:
            raise RuntimeError(f"Scripted transfers stopped: {problem}")
        if completed >= SCHEDULE_STEPS:
            return
        logger.info("Cycle %d: %d of %d schedule steps completed", cycle, completed, SCHEDULE_STEPS)
        run_start(tutorial, **get_strategy_environment(tutorial))
    completed, problem = get_scripted_progress(State.read_json_file(tutorial.state_file), SCHEDULE_STEPS)
    assert completed >= SCHEDULE_STEPS and problem is None, f"Schedule incomplete after {SCHEDULE_STEPS + 1} cycles: {completed} steps done, {problem}"


def verify_state(tutorial: Tutorial) -> None:
    """Check every scripted transfer in the final executor state and print them."""
    run_cli(["check-accounts"], get_cli_environment(tutorial, **get_strategy_environment(tutorial)))
    state = State.read_json_file(tutorial.state_file)
    rows = verify_lighter_scripted_transfers(state, build_lighter_transfer_schedule(get_unit()), tutorial.account_indices)
    print(tabulate(rows, headers="keys", tablefmt="simple"))
    logger.info("All %d scripted transfers were booked correctly", len(rows))


def main() -> None:
    """Run the tutorial phases, skipping those completed in an earlier run."""
    configure_logging()
    config = load_tutorial_config(
        executor_id=EXECUTOR_ID,
        strategy_file=STRATEGY_DIR / "lighter_scripted_transfers_strategy.py",
        subscription_usdc=SCHEDULE_STEPS * get_unit() - Decimal(1),
        sub_account_count=SUB_ACCOUNT_COUNT,
    )
    tutorial = prepare_tutorial(config)
    try:
        run_phases(tutorial, [
            (1, "Deploy the vault with two Lighter sub-accounts", lambda t: deploy(t, "Lighter transfers test", "LTT")),
            (2, "Initialise executor state with one position per account", initialise_state),
            (3, "Subscribe to the vault", lambda t: subscribe(t, **get_strategy_environment(t))),
            (4, "Run the scripted transfer strategy", run_scripted_cycles),
            (5, "Redeem all shares", lambda t: redeem_all_shares(t, drain_cycles=2, **get_strategy_environment(t))),
            (6, "Sweep the Safe and reconcile the deployer's balances", refund),
            (7, "Verify every scripted transfer in the executor state", verify_state),
        ])
    finally:
        tutorial.close()


if __name__ == "__main__":
    main()
