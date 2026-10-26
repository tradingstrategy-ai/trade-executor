# Recover a rejected second-stage HyperCore withdrawal

The 26 October 2026 Tortoise incident moved 3,947.465150 USDC to HyperCore
perp. A second request for 18.336496 USDC was rejected with `Vault deposits
must be at least $5.` despite being a withdrawal. Both HyperEVM receipts
succeeded. Approximately $19.84 remained in the vault. The intended $1.50
remainder may explain the rejection; the protocol boundary is not verified.

## Prevention and limits

Optional same-cycle second withdrawals are disabled pending protocol validation.
The verified first withdrawal continues through the existing transit sweep and
settles as a partial fill; remaining shares stay valued. First-stage withdrawal
rules, NAV safety margins and accounting write-off thresholds are unchanged.

The observed rejection does not establish a universal $5 minimum remainder.
Applying that assumption to first-stage withdrawals would also block ordinary
small closes. Small-position cleanup has not been validated against this failure
and must remain disabled during incident recovery. This change contains the
optional-leg failure; it does not guarantee every future small exit will succeed.

`repair` recognises persisted ambiguous second-stage phases even when a crash
preceded the final stranded marker. It repairs untouched sibling trades and
defers the sell. The scheduler's wrapped known execution halt now requests a
final state save. Arbitrary exceptions do not acquire this behaviour.

## Evidence and command sequence

Use the [production maintenance shell](docker.md#production-maintenance-shell).
Stop the executor, verify its image revision, preserve the JSON state, recorder
files and logs, and confirm the service's existing state/vault/Safe identity.
Use a release containing these changes; v1470 cannot reconcile this sell.

Create a private operator evidence file accessible inside the service mount:

```json
{
  "trade_id": 1825,
  "first_execution_hash": "0x7a430ac1f17f121f7bbc04475e595b00007e22a78c7230f11e0bb614b072ec0a",
  "second_execution_hash": "0x1f558239b1ec0bc120cf04475e5a6e0000909a1f4cef2a93c31e2d8c70efe5ab"
}
```

These are HyperCore **execution** hashes, not EVM or enqueue hashes. The command
checks their Safe, vault, direction, amount, receipt timestamps and successful
first-leg ledger entry. It requires an explicit second execution rejection.
This is a narrow mainnet two-leg withdrawal path, not a general transaction
resumer. Evidence is read-only; the command does not discover candidate actions.

Inside the service maintenance shell:

```shell
poetry run trade-executor repair --id hyper-ai --state-file state/hyper-ai.json
poetry run trade-executor correct-accounts --id hyper-ai --state-file state/hyper-ai.json \
  --dry-run --no-cleanup-hypercore-small-positions \
  --hypercore-withdrawal-evidence state/hyper-ai-withdrawal-evidence.json
```

`repair` changes accounting state and has no dry-run flag. For a rehearsal, use
an isolated state copy and a local RPC fork. It must leave #1825 unresolved and
repair only the untouched siblings #1822–#1824. The correction preview reads
custody and prints the transfer plan without signing, broadcasting, creating a
backup or saving state. Keep small-position cleanup disabled: its normal
preview can construct/sign transactions, so generic `--dry-run` must not be
described universally as a no-signing mode. This preview stops before the generic
correction scan: sale/reserve/slot changes are projections. The live run scans
actual returned balances and may prompt for the reserve correction.

After reviewing and authorising the live transfer plan, run the same
`correct-accounts` command without `--dry-run`. It may transfer before any later
accounting confirmation. Do not use `--skip-save`, `--process-redemption`,
`--transfer-away` or `--skip-hypercore-transit-recovery` for this recovery.

The command reuses the existing perp-to-spot-to-EVM helper and records completed
transfer hashes in the unresolved state. It books the verified partial sale
once through `State.mark_trade_success()`. The transit transfer helper does not credit
reserves; `State.mark_trade_success()` supplies the sale reserve credit. Generic reserve correction handles pre-existing free transit cash;
the original 3.512413 USDC perp balance is not sale profit. Transfer shortfall,
including retained dust, is conservatively deducted from sale proceeds.
Corrections beyond that verified pre-existing reserve cash are refused.

After final account checks pass, sale accounting and consumption of the
26 October partial decision are saved together. The remaining vault shares
stay open. The three stale buys are not submitted and successful historical
trades are not replayed.

If transfers completed and their receipt checkpoints survived but final
accounting did not save, rerun with the same evidence: the command verifies
returned custody and skips transfers. A crash between legs, missing completion
receipts or unexplained balances requires operator reconciliation. Do not
restore old JSON over transactions that already moved funds. This is not an
automatic recovery path for every interruption window.

A second preview with the evidence file reports an already-reconciled trade
without transfers or accounting changes. Run `check-accounts` separately. Check terminal sibling/trade
statuses, valued residual shares, reserve balances and cleared pending decision
before restarting. Verify `executor_running`, exceptions and scheduled progress,
not merely HTTP 200 or a running container. Preserve recorder and original
transaction history throughout.

## Local rehearsal evidence

A fresh remote JSON copy on 26 October had SHA-256
`bc07cedbe132a604f8d976aee6e2be3345b495a3d258ca5d097a6063adacb826`.
It matched the deployment host's hash; its last write was 01:25:47 UTC.

Against disposable copies, the real CLI `repair` and correction preview passed.
A further CLI rehearsal mocked only the HyperCore transfer boundary and credited
the local Anvil Safe: correction and a follow-up preview passed, one recovery
was performed, #1825 became successful with residual shares, cycle advanced to
137, and the pending slot cleared. This used the minimal HyperCore test strategy
for CLI universe construction, not a new Hyper-AI backtest. The real remote
trading loop remained crashed; no recovery transaction was sent to production.
