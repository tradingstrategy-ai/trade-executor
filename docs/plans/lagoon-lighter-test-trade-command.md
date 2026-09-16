# Lagoon Lighter test trade command

## Goal

Provide `trade-executor lagoon-lighter-test-trade` for a small, manual
end-to-end test of an existing Ethereum Lagoon vault with a Safe-owned Lighter
account. It is an operator command, not a strategy execution path.

## Implemented scope

The command:

1. reads the mode-`0600` delegated API-key record created by
   `lagoon-deploy-vault`;
2. checks the Ethereum network, vault, Safe, public Lighter account and
   existing executor state;
3. deposits a bounded USDC amount from the Safe to Lighter;
4. opens and closes a small ETH/USD long;
5. posts Lagoon NAV after the deposit, open, close and final withdrawal;
6. requests one secure Lighter withdrawal, waits for the matching history row
   to become claimable and claims it to the Safe; and
7. persists a private, non-secret recovery journal and lock next to state.

The public Lighter sequencer, order transport and L1 withdrawal proof are
outside Anvil. The Typer integration test therefore uses an Ethereum fixed
block Anvil fork for Lagoon, Safe, USDC, state and NAV, and mocks only those
Lighter-specific operations.

## Configuration

The normal command configuration is used, including `PRIVATE_KEY`,
`JSON_RPC_ETHEREUM`, `ASSET_MANAGEMENT_MODE=lagoon`, `VAULT_ADDRESS`,
`VAULT_ADAPTER_ADDRESS`, `STATE_FILE` and `STRATEGY_FILE`.

Lighter-specific configuration is:

| Variable | Meaning |
| --- | --- |
| `LIGHTER_OPERATOR_RECORD_FILE` | Private delegated-key JSON record. |
| `LIGHTER_ACCOUNT_INDEX` | Optional public cross-check for the record. |
| `LIGHTER_TEST_DEPOSIT_USDC` | Additional Safe USDC deposited for this test, default 20 USDC. |
| `LIGHTER_TEST_POSITION_USDC` | Optional ETH/USD notional. |
| `LIGHTER_TEST_MAX_SLIPPAGE` | Maximum market-order slippage. |
| `LIGHTER_TEST_JOURNAL_FILE` | Optional recovery journal path. |
| `LIGHTER_DEPOSIT_TIMEOUT` | Deposit observation timeout, default 900 seconds. |
| `LIGHTER_POSITION_TIMEOUT` | Position observation timeout, default 300 seconds. |
| `LIGHTER_WITHDRAW_TIMEOUT` | Secure-withdrawal timeout, default 3600 seconds. |

The API private key is read only from the record. It is excluded from object
representations, logs, the state file and the recovery journal.

The 20 USDC default is above Lighter's 1 USDC direct Ethereum contract deposit
minimum and provides headroom over the observed ETH/USD limits of 0.005 ETH and
10 USDC notional. The command reads current market metadata before sizing the
order because these limits and the ETH price can change.

## Recovery rules

The compact journal phases are:

```text
created -> deposited -> opened -> closed -> withdrawal_requested -> withdrawn -> complete
```

Before each run, the command checks the journal against the public account and
Safe balance. It rejects negative Safe or Lighter values and never opens or
closes an unexplained position.

The deposit and withdrawal SDK calls are not replay-safe. The command refuses
to repeat a deposit if either the Safe balance or Lighter collateral differs
from the recorded baseline. It records withdrawal intent before the SDK call;
if that call was not accepted, the operator must confirm that withdrawal
history has no matching row, reset the journal phase to `closed`, then retry.

Lighter's dynamic `withdrawalDelay` is logged before a secure withdrawal. It is
only an estimate: the command waits for a matching `withdraw_history` row to
become `claimable`. A small timestamp tolerance permits local/API clock skew
while excluding older equal-value withdrawals.

Fast withdrawal is intentionally unsupported. A Safe owns the Lighter account
and does not possess the L1 EOA private key required for Lighter fast
withdrawal.

## Verification

Run the focused checks:

```shell
source .local-test.env
PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest \
  tests/lagoon/test_lagoon_lighter_test_trade.py::test_cli_lagoon_lighter_test_trade_resumes_after_lighter_failure
PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest \
  deps/web3-ethereum-defi/tests/lighter/test_lighter_api.py::test_fetch_lighter_withdrawal_delay
```

The first test asserts secret redaction, the single-chain deployment artefact,
resume after a mocked Lighter order failure, final Safe balance and zero-valued
external account position. The public endpoint test confirms that the current
withdrawal delay is positive.
