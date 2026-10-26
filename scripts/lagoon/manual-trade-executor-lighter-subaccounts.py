"""Tutorial: run a Lagoon vault with a Lighter master account and two sub-accounts.

This live Ethereum mainnet tutorial exercises the multi-account custody
lifecycle with real money, driven by the production cash policy of
``strategies/test_only/lighter_multi_subaccount_strategy.py``:

1. Deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, each sub-account with its own API key for an external trader
2. Initialise executor state with one position per Lighter account
3. Subscribe to the vault and let ``start`` deposit the idle Safe cash to the master
4. Let ``start`` fund both sub-accounts to their declared 50/50 weights
5. Act as the external trader: transfer USDC from sub-account 1 to
   sub-account 2 with sub-account 1's own API key, and check that the next
   ``start`` cycle revalues both positions
6. Let ``start`` withdraw the now overweight sub-account 2 to the Safe with
   the key-free L1 ``withdraw``
7. Let ``start`` redeposit the withdrawal to the master
8. Let ``start`` fund the underweight sub-account 1
9. Request a full redemption, let ``start`` drain every account to the Safe,
   and redeem all shares with ``lagoon-redeem``
10. Sweep the remaining Safe USDC and ETH back to the deployer and print a
    reconciliation table
11. Print every trade with its description, status and transaction hashes

Every executor custody movement runs through the real ``start`` command and
the production ``decide_trades()`` policy; the tutorial never changes the
strategy between cycles. No orders are placed. The executor holds only the
master key, so it cannot make the trader's sub-account to sub-account
transfer; that transfer is what makes the policy rebalance. For a tutorial in
which the executor signs every movement itself, including transfers between
sub-accounts, and which can also run on an Anvil fork, see
``manual-trade-executor-lighter-transfers.py``.

An earlier form of this tutorial ran end to end on 2026-10-26; the trader
transfer step uses the real Lighter SDK, so this script supports Ethereum
mainnet only. A master secure withdrawal takes about 20 minutes and a
sub-account L1 withdrawal took about 29 minutes.

Funding: the deployer needs 5 USDC plus mainnet gas ETH (about 0.011 ETH in
the 2026-10-26 run). Deployment subscribes 1 USDC and moves it to the master
account; the tutorial subscribes 4 USDC more. Lighter's transfer, secure
withdrawal and direct deposit minimums are 1 USDC, and every movement here
is exactly at or above them: the master funds each sub-account with 2.5 USDC,
the trader moves 1 USDC, and the policy then withdraws, redeposits and
transfers 1 USDC. Lighter charged nothing for same-owner transfers in the
2026-10-26 run and all 5 USDC returned to the deployer.

Secrets, key backups during deployment and resuming from ``progress.json``
work as described in :py:mod:`tradeexecutor.testing.lighter_tutorial`. If a
``start`` cycle was interrupted during a transfer, the next ``start`` refuses
to run; recover it with ``trade-executor check-accounts`` and
``trade-executor correct-accounts`` using the environment printed in the
log, then rerun the tutorial.

Example
-------

.. code-block:: shell

    source .local-test.env
    export LIGHTER_TEST_PRIVATE_KEY="0x..."
    export JSON_RPC_ETHEREUM="https://..."
    poetry run python scripts/lagoon/manual-trade-executor-lighter-subaccounts.py

``LIGHTER_RUN_DIR`` selects the private working directory to resume from.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any

from eth_defi.lighter.sdk import create_lighter_signer_client
from eth_defi.lighter.transfer import create_lighter_auth_token_manager, fetch_lighter_transfer_fee, request_lighter_transfer, wait_for_lighter_transfer

from tradeexecutor.state.state import State
from tradeexecutor.testing.lighter_report import print_trade_summary
from tradeexecutor.testing.lighter_tutorial import (
    STRATEGY_DIR,
    Tutorial,
    configure_logging,
    deploy,
    fetch_lighter_balances,
    initialise_state,
    load_tutorial_config,
    prepare_tutorial,
    redeem_all_shares,
    refund,
    run_phases,
    run_start,
    subscribe,
)

logger = logging.getLogger(__name__)

#: Executor id and artefact prefix of this tutorial.
EXECUTOR_ID = "lighter-subaccounts"
#: Subscription after the 1 USDC activation: the master funds each sub-account with 2.5 USDC.
SUBSCRIPTION_USDC = Decimal(4)
#: USDC the trader moves from sub-account 1 to sub-account 2, Lighter's transfer minimum.
TRADER_TRANSFER_USDC = Decimal(1)


def subscribe_and_deposit(tutorial: Tutorial) -> None:
    """Subscribe to Lagoon; the same ``start`` cycle deposits the idle Safe cash to the master."""
    subscribe(tutorial)
    assert tutorial.usdc.fetch_balance_of(tutorial.safe_address) == 0, "The policy left cash in the Safe"


async def submit_trader_transfer(tutorial: Tutorial, handover: dict[str, Any], to_account_index: int) -> None:
    """Sign a sub-account to sub-account transfer with the sending sub-account's own key.

    A submission is not idempotent, so its hash is saved to ``progress.json``
    before waiting; a resumed run confirms the saved hash instead of sending
    the transfer again.
    """
    from_account_index = int(handover["account_index"])
    api_key_index = int(handover["api_key_index"])
    client = create_lighter_signer_client(from_account_index, api_key_index, handover["private_key"])
    try:
        auth = create_lighter_auth_token_manager(client, api_key_index)
        tx_hash = tutorial.progress.get("trader_transfer_tx_hash")
        if tx_hash is None and tutorial.progress.get("trader_transfer_submitting"):
            raise RuntimeError(
                f"A transfer from sub-account {from_account_index} may have been submitted without a recorded hash; "
                f"check its transfer history before rerunning"
            )
        if tx_hash is None:
            tutorial.progress["trader_transfer_submitting"] = True
            tutorial.write_progress()
            fee = await fetch_lighter_transfer_fee(client, auth, to_account_index=to_account_index)
            logger.info("Lighter quotes a %s USDC fee from sub-account %d to sub-account %d", fee, from_account_index, to_account_index)
            tx_hash = await request_lighter_transfer(client, to_account_index=to_account_index, amount=TRADER_TRANSFER_USDC, fee=fee, api_key_index=api_key_index)
            tutorial.progress["trader_transfer_tx_hash"] = tx_hash
            tutorial.write_progress()
        record = await wait_for_lighter_transfer(client, auth, account_index=from_account_index, tx_hash=tx_hash, timeout=tutorial.config.transfer_timeout)
    finally:
        await client.close()
    assert (record.from_account_index, record.to_account_index, record.amount) == (from_account_index, to_account_index, TRADER_TRANSFER_USDC), f"Unexpected transfer record {record}"
    tutorial.progress["trader_transfer_fee_usdc"] = str(record.fee)
    logger.info("Trader transfer %s executed: %s USDC, fee %s USDC", tx_hash, record.amount, record.fee)


def transfer_between_sub_accounts(tutorial: Tutorial) -> None:
    """Move USDC from sub-account 1 to sub-account 2 as the external trader would.

    The executor's positions are stale until the next ``start`` cycle
    revalues each account from Lighter. That cycle also starts the policy's
    rebalance, as sub-account 2 is now overweight.
    """
    _master, sub_1, sub_2 = tutorial.account_indices
    # The key is read into memory only and never logged.
    handover = json.loads(tutorial.get_handover_file(tutorial.sub_account_ids[0]).read_text())
    assert int(handover["account_index"]) == sub_1, f"Hand-over file is not for sub-account {sub_1}"

    # Save the balances before submitting, so a resumed run compares against them.
    if "trader_transfer_before" not in tutorial.progress:
        tutorial.progress["trader_transfer_before"] = {str(index): str(value) for index, value in fetch_lighter_balances(tutorial).items()}
        tutorial.write_progress()
    before = {int(index): Decimal(value) for index, value in tutorial.progress["trader_transfer_before"].items()}
    asyncio.run(submit_trader_transfer(tutorial, handover, sub_2))

    after = fetch_lighter_balances(tutorial)
    fee = Decimal(tutorial.progress["trader_transfer_fee_usdc"])
    assert after[sub_2] == before[sub_2] + TRADER_TRANSFER_USDC, f"Sub-account {sub_2} did not receive the transfer"
    assert sum(after.values(), Decimal(0)) == sum(before.values(), Decimal(0)) - fee, "The transfer changed total Lighter equity by more than its fee"


def rebalance(tutorial: Tutorial, expected: dict[str, Decimal]) -> None:
    """Run one ``start`` cycle and check the executor tracks every account at its Lighter equity."""
    run_start(tutorial)
    state = State.read_json_file(tutorial.state_file)
    tracked = {
        int(position.pair.get_exchange_account_id()): Decimal(str(position.get_quantity()))
        for position in state.portfolio.open_positions.values()
        if position.is_exchange_account()
    }
    balances = fetch_lighter_balances(tutorial)
    for index, value in balances.items():
        assert tracked[index] == value, f"Executor tracks account {index} at {tracked[index]}, Lighter reports {value}"
    accounts = tutorial.get_accounts()
    for account_id, value in expected.items():
        assert balances[accounts[account_id]] == value, f"Expected {account_id} to hold {value} USDC, Lighter reports {balances[accounts[account_id]]}"


def main() -> None:
    """Run the tutorial phases, skipping those completed in an earlier run."""
    configure_logging()
    config = load_tutorial_config(executor_id=EXECUTOR_ID, strategy_file=STRATEGY_DIR / "lighter_multi_subaccount_strategy.py", subscription_usdc=SUBSCRIPTION_USDC)
    if config.simulate:
        raise RuntimeError("This tutorial is live Ethereum/Lighter only; run manual-trade-executor-lighter-transfers.py with SIMULATE=true instead")
    tutorial = prepare_tutorial(config)
    try:
        run_phases(tutorial, [
            (1, "Deploy the vault with two Lighter sub-accounts", lambda t: deploy(t, "Lighter sub-accounts test", "LSA")),
            (2, "Initialise executor state with one position per account", initialise_state),
            (3, "Subscribe and deposit idle Safe cash to the master", subscribe_and_deposit),
            (4, "Fund both sub-accounts from the master", lambda t: rebalance(t, {"master": Decimal(0), "trend": Decimal("2.5"), "arb": Decimal("2.5")})),
            (5, "Transfer from sub-account 1 to sub-account 2 as the external trader", transfer_between_sub_accounts),
            (6, "Withdraw the overweight sub-account 2 to the Safe with the key-free L1 path", lambda t: rebalance(t, {"trend": Decimal("1.5"), "arb": Decimal("2.5")})),
            (7, "Redeposit the withdrawal to the master", lambda t: rebalance(t, {"master": Decimal(1)})),
            (8, "Fund the underweight sub-account 1 from the master", lambda t: rebalance(t, {"master": Decimal(0), "trend": Decimal("2.5"), "arb": Decimal("2.5")})),
            (9, "Drain every account and redeem all shares", lambda t: redeem_all_shares(t, drain_cycles=2)),
            (10, "Sweep the Safe and reconcile the deployer's balances", lambda t: refund(t, extra_fees_usdc=Decimal(t.progress.get("trader_transfer_fee_usdc", "0")))),
            (11, "Print every trade with its description and transaction hashes", lambda t: print_trade_summary(State.read_json_file(t.state_file))),
        ])
    finally:
        tutorial.close()


if __name__ == "__main__":
    main()
