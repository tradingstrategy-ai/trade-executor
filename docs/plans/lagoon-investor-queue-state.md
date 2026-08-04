# Lagoon investor queue state export

## Objective

Expose the current unsettled Lagoon investor deposit and redemption amounts in
the persistent executor state returned by the state API.

GuardV0's automatic-settlement cap and cooldown remain static/live policy
information under `/metadata`. The investor queue is mutable treasury state and
must instead be stored under `state.sync.treasury`, where Lagoon already records
the pending redemption amount used by strategy allocation and yield management.

The change must keep these meanings distinct:

| Value | Location | Meaning |
|---|---|---|
| Guard cap and cooldown | `metadata.on_chain_data.smart_contracts.lagoon_guard_v0` | On-chain policy controlling automatic settlement |
| Pending deposits | `state.sync.treasury.pending_deposits` | Underlying assets still held in the Lagoon pending Silo |
| Pending redemptions | `state.sync.treasury.pending_redemptions` | Estimated underlying assets needed for redemption shares still held in the Silo |

## Existing behaviour

`Treasury.pending_redemptions` already stores the underlying-denominated
redemption queue so strategies can retain enough liquid cash. There is no
corresponding pending-deposit field.

`LagoonVaultSyncModel.sync_treasury()` already has the correct lifecycle
boundaries:

1. reconcile the Safe reserve and calculate NAV;
2. post and confirm the NAV transaction;
3. inspect the Silo balances and run GuardV0 settlement preflight;
4. either leave the queue deferred or broadcast settlement; and
5. update treasury sync metadata.

For a deferred, manual or empty queue, the method currently refreshes pending
redemptions and share count. After successful settlement it uses the settlement
analysis to refresh pending redemptions and share count. Pending deposits are
not persisted in either branch.

## State model

Add the following optional field to `tradeexecutor.state.sync.Treasury`:

```python
pending_deposits: USDollarAmount | None = None
```

Keep the existing `pending_redemptions` field and its public semantics. Both
values are human-readable underlying-token amounts. Lagoon strategies currently
require a stablecoin reserve, so these are also treated as US dollar amounts by
the existing treasury API.

The value distinction is intentional:

- `None` means the queue has not been observed, or the sync model does not
  provide this Lagoon-specific information;
- `0.0` means the queue was observed on-chain and was empty at the recorded
  treasury block.

Do not put the queue in `Portfolio`, reserve positions or `Metadata`. An
unsettled deposit is still outside the Safe and does not belong to strategy NAV.
A pending redemption is a liquidity requirement, not a reduction in portfolio
ownership until settlement occurs.

Do not create a `BalanceUpdate` merely because the queue snapshot changes.
Balance updates continue to represent assets actually settled into or out of
the strategy treasury.

Adding a defaulted field keeps older state JSON readable. The normal state
serialiser will expose the new field automatically; no webhook endpoint or
response adapter is needed.

## Queue snapshot timing

Persist the queue only from a completed `sync_treasury(post_valuation=True)`
path whose NAV transaction has been confirmed. Redemption shares are converted
to underlying at the post-NAV share price, so a pre-NAV estimate can be stale.

Use one explicit block identifier for every value in a snapshot and record that
same block in `Treasury.last_block_scanned`. Avoid mixing a `latest` deposit read
with a redemption conversion from another block.

### Deferred, manual and empty queues

After NAV confirmation and GuardV0 preflight determines that no settlement will
be broadcast:

1. choose the current confirmed snapshot block;
2. read pending underlying deposits from the Silo at that block;
3. calculate pending redemptions in underlying at that block;
4. read total share supply at that block; and
5. update `pending_deposits`, `pending_redemptions`, `share_count`, timestamps
   and `last_block_scanned` together through `_mark_treasury_sync_completed()`.

This covers an empty queue, a cooldown deferral and an oversized queue requiring
direct Safe governance. In all three cases state reflects what remains
unsettled, even though the method returns no `BalanceUpdate`.

Update `_mark_treasury_sync_completed()` to accept `pending_deposits` alongside
the existing pending-redemption and share-count values. Passing an observed zero
must write zero rather than retaining a stale non-zero value.

Make the signature change explicit so every caller is audited:

```python
def _mark_treasury_sync_completed(
    self,
    treasury_sync,
    strategy_cycle_ts: datetime.datetime,
    block_number: int,
    pending_deposits: Decimal | None = None,
    pending_redemptions: Decimal | None = None,
    share_count: Decimal | None = None,
) -> None:
```

The no-settlement branch currently records the NAV receipt block while reading
queue values later. Replace that with the queue snapshot block. The two blocks
may be equal, but state must never claim the NAV block when its queue values
were actually sampled at a later block.

### Successful settlement

After the settlement receipt is confirmed and analysed:

1. use the settlement/analysis block as the snapshot block;
2. read the remaining pending deposit balance at that exact block;
3. calculate the remaining pending redemption amount at that exact block;
4. update both queue fields together with share count and existing treasury
   timestamps; and
5. keep creation of the reserve `BalanceUpdate` unchanged.

Stock Lagoon normally clears the deposit queue completely, but the explicit
post-settlement read prevents a stale preflight amount from being exported and
documents the state as an as-of-block snapshot.

Calls with `post_valuation=False`, disabled broadcasting, failed NAV posting or
an unexpected settlement error must not claim to have produced a fresh
post-NAV queue snapshot. Preserve their existing early-return or failure
semantics.

## Integration testing

Extend the existing Base Anvil-fork GuardV0 integration coverage in
`tests/lagoon/test_lagoon_guard_settlement.py`; do not add another expensive
deployment fixture or a separate deployment-heavy test.

In
`test_lagoon_guard_automatically_settles_flow_below_settlement_limit()`:

1. before the first treasury sync, assert `pending_deposits is None` and that
   state JSON serialises the unobserved value as `null`;
2. after the initial 9 USDC deposit settles, assert
   `pending_deposits == 0` and `pending_redemptions == 0`;
3. after the second 1 USDC deposit is deferred by cooldown, assert
   `pending_deposits == 1` and `pending_redemptions == 0`;
4. serialise the `State`, inspect the JSON path
   `sync.treasury.pending_deposits`, and round-trip it through
   `State.read_json_blob()` to prove the value is part of persisted state rather
   than frontend metadata; and
5. retain the existing metadata assertions to demonstrate that policy and queue
   data remain in their separate API objects.

These assertions cover the `None` to observed-zero transition without adding a
separate unit test or another deployment.

Also strengthen the existing mixed-flow assertion in
`test_lagoon_guard_posts_nav_but_defers_oversized_flow_to_safe()`:

1. assert the preserved 9 USDC Silo deposit is exported as
   `pending_deposits == 9`;
2. retain the existing approximately 2 USDC `pending_redemptions` assertion;
3. assert `last_block_scanned` equals the chain block at which the preserved
   queue was read after the NAV-only sync, rather than an earlier pre-NAV or NAV
   receipt block; and
4. retain the proof that no settlement transaction or balance update occurred.

Use `pytest.approx()` for both money values. Update the ordered test docstrings
and matching numbered body comments when adding the assertions.

## Documentation updates

Update `.claude/docs/lagoon-treasury-settlement.md` with a persisted queue-state
section covering:

- the two state JSON paths and their underlying-token units;
- `None` versus an observed zero;
- the post-NAV/preflight snapshot for deferred queues;
- the post-settlement refresh for successful queues;
- `last_block_scanned` as the queue snapshot block, including why it advances
  even when no settlement transaction and no balance update occurs;
- why queue changes do not create balance updates or enter NAV; and
- the boundary between dynamic `/state` queue data and Guard policy in
  `/metadata`.

Update the scope note in `.claude/docs/vault-deposit-redeem.md` to name
`state.sync.treasury.pending_deposits` and `pending_redemptions` as the
strategy-owned Lagoon investor queue. This prevents these fields from being
confused with asynchronous positions where the strategy is itself depositing
into an external ERC-7540 vault.

Add a concise `CHANGELOG.md` entry when preparing the feature pull request,
using the pull request date required by the repository conventions.

## Verification

Prepare the copied worktree environment and run the two focused integration
tests separately with the required extended timeout:

```shell
source .local-test.env
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lagoon_guard_settlement.py::test_lagoon_guard_automatically_settles_flow_below_settlement_limit
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/lagoon/test_lagoon_guard_settlement.py::test_lagoon_guard_posts_nav_but_defers_oversized_flow_to_safe
```

Do not run the full test suite for this focused state-export change.

## Acceptance criteria

- Lagoon state JSON exposes current unsettled deposits and redemptions under
  `sync.treasury`.
- Both values are sampled at, and traceable to, one recorded on-chain block.
- A successful settlement overwrites the pre-settlement deposit amount with
  the remaining post-settlement amount, normally zero.
- Cooldown and oversized queues retain their non-zero state values without a
  false `BalanceUpdate`.
- Observed empty queues serialise as zero; an unobserved/non-Lagoon queue remains
  `None`.
- Existing GuardV0 policy metadata remains unchanged and separate from the
  queue state.
- Existing state files lacking `pending_deposits` continue to load.
- The existing fork tests cover settlement, cooldown deferral, mixed oversized
  flow and state serialisation without introducing a new deployment-heavy test.

## Out of scope

- Moving GuardV0 limit or cooldown fields out of metadata.
- Adding per-investor queue entries; only aggregate Silo amounts are exported.
- Adding pending deposits to strategy NAV or available cash.
- Changing settlement eligibility, GuardV0 preflight or Safe-governance
  recovery behaviour.
- Emitting historical queue events or charts; this change stores only the
  latest treasury snapshot.
