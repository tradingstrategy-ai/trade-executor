"""Tutorial: run a Lagoon vault with a Lighter master account and two sub-accounts.

This live Ethereum mainnet tutorial exercises the multi-account custody
lifecycle with real money:

1. Deploy a Lagoon vault whose Safe owns a Lighter master account and two
   sub-accounts, each sub-account with its own API key for an external trader
2. Initialise executor state with one position per Lighter account
3. Subscribe to the vault and let ``start`` deposit the idle Safe cash to the master
4. Fund both sub-accounts from the master
5. Withdraw sub-account 2 to the Safe with the key-free L1 ``withdraw``
6. Redeposit the withdrawal to the master
7. Fund sub-account 1 from the master
8. Act as the external trader: transfer USDC from sub-account 1 to
   sub-account 2 with sub-account 1's own API key, and check that the next
   ``start`` cycle revalues both positions
9. Request a full redemption, let ``start`` drain every account to the Safe,
   and redeem all shares with ``lagoon-redeem``
10. Sweep the remaining Safe USDC and ETH back to the deployer and print a
    reconciliation table

Every executor custody movement runs through the real ``start`` command and
the production ``decide_trades()`` helper, with
``strategies/test_only/lighter_multi_subaccount_strategy.py``. No orders are
placed. The executor holds only the master key, so it cannot make the
trader's sub-account to sub-account transfer; it only revalues the positions.
For a tutorial in which the executor signs every movement itself, including
transfers between sub-accounts, and which can also run on an Anvil fork, see
``manual-trade-executor-lighter-transfers.py``.

The tutorial ran end to end on 2026-10-26 before step 8 was added; step 8
has not yet run live. Step 8 uses the real Lighter SDK, so this script
supports Ethereum mainnet only. A master secure withdrawal takes about 20
minutes and a sub-account L1 withdrawal took about 29 minutes.

Funding: the deployer needs 5 USDC plus mainnet gas ETH (about 0.011 ETH in
the 2026-10-26 run). Deployment subscribes 1 USDC and moves it to the master
account; the tutorial subscribes 4 USDC more. Lighter's transfer, secure
withdrawal and direct deposit minimums are 1 USDC. The strategy budgets the
1 USDC fee that Lighter's public endpoint quotes on top of each master to
sub-account transfer, so every movement still reaches the minimum if it is
charged. Lighter charged nothing for same-owner transfers in the 2026-10-26
run and all 5 USDC returned to the deployer.

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

Optional environment variables
------------------------------

``LIGHTER_SUBSCRIPTION_USDC``
    USDC subscribed after deployment. Defaults to and must be at least 4.

``LIGHTER_API_KEY_INDEX``
    API-key slot for the master and the sub-account keys. Defaults to 4.

``LIGHTER_RUN_DIR``
    Private working directory. Defaults below ``~/.tradingstrategy/examples``.

``LIGHTER_WITHDRAW_TIMEOUT`` / ``LIGHTER_TRANSFER_TIMEOUT``
    Maximum seconds to wait for a withdrawal and a sub-account transfer.
    Default 3600 and 300.
"""

import asyncio
import json
import logging
import os
from decimal import Decimal
from typing import Any

from eth_defi.lighter.sdk import create_lighter_signer_client
from eth_defi.lighter.transfer import create_lighter_auth_token_manager, fetch_lighter_transfer_fee, request_lighter_transfer, wait_for_lighter_transfer

from tradeexecutor.state.state import State
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
#: Number of sub-accounts the tutorial creates.
SUB_ACCOUNT_COUNT = 2
#: USDC the strategy keeps in the Safe; none, so all capital moves to Lighter.
SAFE_BUFFER_USDC = Decimal(0)
#: Fee budgeted on top of each master to sub-account transfer: the public
#: ``transferFeeInfo`` quote. The authenticated quote, and the fee charged
#: between same-owner accounts, was 0 on 2026-10-26.
TRANSFER_FEE_USDC = Decimal(1)
#: Smallest subscription for which every movement reaches Lighter's 1 USDC
#: minimum even when the budgeted fee is charged. With the 1 USDC bootstrap
#: the master holds 5 USDC: two 1 USDC fundings and their fees leave 1 USDC,
#: the redeposited 1 USDC withdrawal makes 2 USDC, and the last funding moves
#: 1 USDC plus its fee.
MIN_SUBSCRIPTION_USDC = Decimal(4)
#: Funding weights: a fifth of the capital, 1 USDC at the minimum, per sub-account.
FUNDING_WEIGHTS = "0.2,0.2"
#: Rebalance weights: drain sub-account 2 and move its capital to sub-account 1.
REBALANCE_WEIGHTS = "1,0"
#: USDC the trader moves from sub-account 1 to sub-account 2, Lighter's transfer minimum.
TRADER_TRANSFER_USDC = Decimal(1)


def get_strategy_environment(weights: str = "") -> dict[str, str]:
    """Return the cash-management strategy settings for one ``start`` cycle."""
    return {
        "LIGHTER_SUB_ACCOUNT_WEIGHTS": weights,
        "LIGHTER_SAFE_CASH_BUFFER_USD": str(SAFE_BUFFER_USDC),
        "LIGHTER_TRANSFER_FEE_USD": str(TRANSFER_FEE_USDC),
    }


def rebalance(tutorial: Tutorial, weights: str) -> None:
    """Run one ``start`` cycle towards the given sub-account weights."""
    run_start(tutorial, **get_strategy_environment(weights))


def subscribe_and_deposit(tutorial: Tutorial) -> None:
    """Subscribe to Lagoon; the same ``start`` cycle deposits the idle Safe cash to the master."""
    subscribe(tutorial, **get_strategy_environment())
    assert tutorial.usdc.fetch_balance_of(tutorial.progress["safe_address"]) == SAFE_BUFFER_USDC


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
    revalues each account from Lighter. That cycle has no target weights, so
    it moves no capital.
    """
    _master_index, sub_1, sub_2 = tutorial.account_indices
    # The key is read into memory only and never logged.
    handover = json.loads(tutorial.get_handover_file(sub_1).read_text())
    assert int(handover["account_index"]) == sub_1, f"Hand-over file is not for sub-account {sub_1}"

    # Save the balances before submitting, so a resumed run compares against them.
    if "trader_transfer_before" not in tutorial.progress:
        tutorial.progress["trader_transfer_before"] = {str(index): str(value) for index, value in fetch_lighter_balances(tutorial).items()}
        tutorial.write_progress()
    before = {int(index): Decimal(value) for index, value in tutorial.progress["trader_transfer_before"].items()}
    asyncio.run(submit_trader_transfer(tutorial, handover, sub_2))

    rebalance(tutorial, "")
    after = fetch_lighter_balances(tutorial)
    fee = Decimal(tutorial.progress["trader_transfer_fee_usdc"])
    assert after[sub_2] == before[sub_2] + TRADER_TRANSFER_USDC, f"Sub-account {sub_2} did not receive the transfer"
    assert sum(after.values(), Decimal(0)) == sum(before.values(), Decimal(0)) - fee, "The transfer changed total Lighter equity by more than its fee"

    # The executor must now track each account at its Lighter equity.
    state = State.read_json_file(tutorial.state_file)
    tracked = {
        int(position.pair.get_exchange_account_id()): Decimal(str(position.get_quantity()))
        for position in state.portfolio.open_positions.values()
        if position.is_exchange_account()
    }
    for index, value in after.items():
        assert tracked[index] == value, f"Executor tracks account {index} at {tracked[index]}, Lighter reports {value}"


def main() -> None:
    """Run the tutorial phases, skipping those completed in an earlier run."""
    configure_logging()
    subscription = Decimal(os.environ.get("LIGHTER_SUBSCRIPTION_USDC", str(MIN_SUBSCRIPTION_USDC)))
    if subscription < MIN_SUBSCRIPTION_USDC:
        raise RuntimeError(f"LIGHTER_SUBSCRIPTION_USDC must be at least {MIN_SUBSCRIPTION_USDC} for every movement to reach Lighter's 1 USDC minimum")
    config = load_tutorial_config(
        executor_id=EXECUTOR_ID,
        strategy_file=STRATEGY_DIR / "lighter_multi_subaccount_strategy.py",
        subscription_usdc=subscription,
        sub_account_count=SUB_ACCOUNT_COUNT,
    )
    if config.simulate:
        raise RuntimeError("This tutorial is live Ethereum/Lighter only; run manual-trade-executor-lighter-transfers.py with SIMULATE=true instead")
    tutorial = prepare_tutorial(config)
    try:
        run_phases(tutorial, [
            (1, "Deploy the vault with two Lighter sub-accounts", lambda t: deploy(t, "Lighter sub-accounts test", "LSA")),
            (2, "Initialise executor state with one position per account", initialise_state),
            (3, "Subscribe and deposit idle Safe cash to the master", subscribe_and_deposit),
            (4, "Fund both sub-accounts from the master", lambda t: rebalance(t, FUNDING_WEIGHTS)),
            (5, "Withdraw sub-account 2 to the Safe with the key-free L1 path", lambda t: rebalance(t, REBALANCE_WEIGHTS)),
            (6, "Redeposit the withdrawal to the master", lambda t: rebalance(t, REBALANCE_WEIGHTS)),
            (7, "Fund sub-account 1 from the master", lambda t: rebalance(t, REBALANCE_WEIGHTS)),
            (8, "Transfer from sub-account 1 to sub-account 2 as the external trader", transfer_between_sub_accounts),
            (9, "Drain every account and redeem all shares", lambda t: redeem_all_shares(t, drain_cycles=2, **get_strategy_environment())),
            (10, "Sweep the Safe and reconcile the deployer's balances", lambda t: refund(t, extra_fees_usdc=Decimal(t.progress.get("trader_transfer_fee_usdc", "0")))),
        ])
    finally:
        tutorial.close()


if __name__ == "__main__":
    main()
