# Recover a rejected second-stage HyperCore withdrawal

The 26 October 2026 Tortoise incident moved 3,947.465150 USDC to HyperCore
perp. A second request for 18.336496 USDC was rejected with `Vault deposits
must be at least $5.` despite being a withdrawal. Both HyperEVM receipts
succeeded. Approximately $19.84 remained in the vault. The intended $1.50
remainder may explain the rejection; the protocol boundary is not verified.

## Prevention and limits

Optional same-cycle second withdrawals are disabled pending protocol validation.
The verified first withdrawal continues through the existing transit sweep and
settles as a partial fill; remaining shares stay valued.

On 28 October 2026 the next cycle closed the $19.84 residual itself. Its first
and only withdrawal, 18.336430 USDC leaving $1.50, was rejected with the same
`Vault deposits must be at least $5.` error and no USDC moved. Earlier closes
leaving $1.50 had succeeded, so the remainder rule is observed, not documented.
A normal full close now leaves at least `HYPERCORE_FULL_CLOSE_MIN_REMAINDER_USD`
($5 minimum plus the $1.50 NAV-drift floor) in the vault, and the accepted
residual boundary rises to $8 so that such a close still settles as closed. A
close whose equity cannot fund a verifiable withdrawal above the remainder is
expired before execution and the position stays open. Small-position cleanup
keeps its 0.10 USDC margin, has not been validated against this rejection and
must remain disabled.

A phase-1 timeout still records the request as stranded perp USDC, so the
executor stops fail-closed. `repair` clears that marker only for a persisted
phase-1 perp-wait failure whose vault equity still matches the pre-phase-1
snapshot and whose Safe has no HyperCore `vaultWithdraw` ledger row from that
vault since the trade started. It then repairs the failed sell as an ordinary
failure with no funds moved, and the restart guard skips the expired slot.
Check the perp balance before `correct-accounts`: unrelated vault distributions
can land there and must not be attributed to the rejected sell.

`repair` recognises persisted ambiguous second-stage phases even when a crash
preceded the final stranded marker. It repairs untouched sibling trades and
defers the sell. The scheduler's wrapped known execution halt now requests a
final state save. Arbitrary exceptions do not acquire this behaviour.

## Evidence and command sequence

Run the commands as [production CLI commands](docker.md#production-cli-commands).
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

From `~/strategies`, with the service stopped:

```shell
docker compose run --rm --no-deps hyper-ai repair
docker compose run --rm --no-deps hyper-ai correct-accounts \
  --dry-run --no-cleanup-hypercore-small-positions \
  --hypercore-withdrawal-evidence state/hyper-ai-withdrawal-evidence.json
```

The 28 October rejected close needs no evidence file. Run `repair`, then the
preview without `--hypercore-withdrawal-evidence`:

```shell
docker compose run --rm --no-deps hyper-ai repair
docker compose run --rm --no-deps hyper-ai correct-accounts --dry-run --no-cleanup-hypercore-small-positions
docker compose run --rm --no-deps hyper-ai correct-accounts --no-cleanup-hypercore-small-positions \
  && docker compose run --rm --no-deps hyper-ai check-accounts \
  && docker compose up -d --no-deps --force-recreate hyper-ai
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

### Recovery stopped before signing on 26 October

The initial recovery implementation omitted wallet nonce initialisation. Its
first transfer could fail with `Nonce is not yet synced from the blockchain`
after saving an empty recovery checkpoint, but before signing or broadcasting.
The hotfix initialises the wallet before saving a new checkpoint.

For that specific legacy checkpoint only, an operator who has verified the
complete traceback proves failure before the first signing/broadcast attempt
may add `"recovery_failed_before_signing": true` to the evidence JSON. Keep both
execution hashes and `trade_id` unchanged. Run the same preview first, then the
live command. This requires a release containing the nonce hotfix.

This field is an operator attestation, not chain-derived proof. Never use it
for a timeout, missing receipt, unknown broadcast outcome or a later retry.
The command accepts only the old checkpoint containing `expected_return` and
an empty `completed` mapping, rechecks the original custody balances, and
retains the old checkpoint in the new attempt's audit. New attempt checkpoints
cannot use this override. Do not delete the checkpoint or restore an old state.
The retry also refuses a wallet with a different pending nonce. This RPC check
does not replace the operator's review of the original signing failure.
Failures during call construction or gas estimation can still occur after a
new checkpoint is saved; these require operator investigation, not this override.

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
