# Plutus lockup unit fix plan

## Goal

Fix `trade-ui` showing Plutus Hedge lockup as millions of days, e.g.
`~2592000.0d`, when the intended lockup is about 30 days.

The bug affects any vault metadata path where the upstream `lockup` field is a
serialised `datetime.timedelta` in seconds but downstream code treats it as
days.

## Current findings

- eth_defi scans ERC-4626 vault lockups through `vault.get_estimated_lock_up()`.
- `eth_defi.erc_4626.scan` asserts the returned value is `datetime.timedelta`.
- `eth_defi.research.vault_metrics.export_lifetime_row()` serialises
  `datetime.timedelta` and `pd.Timedelta` values using `value.total_seconds()`.
- A 30 day Plutus lockup is therefore exported as `2592000.0`.
- `tradingstrategy.alternative_data.vault._parse_vault_metadata()` assigns
  `VaultMetadata(lockup_days=entry.get("lockup"))` without converting seconds
  to days.
- trade-executor PR `#1535` copies `VaultMetadata.lockup_days` to
  `pair.other_data["vault_lockup_days"]`, displays it as days, and uses it in
  `datetime.timedelta(days=lockup_days)`.

Reproduction with current code:

- `pair.other_data["vault_lockup_days"] = 2592000.0` renders as
  `~2592000.0d`.
- A successful deposit with that value stores
  `vault_lockup_expires_at = "9123-01-27T00:00:00"` for a 2026-06-01 deposit.

## Fix strategy

Fix the data contract at the source while keeping trade-executor defensive
enough to handle already exported metadata.

The canonical meaning should be:

- `lockup`: legacy top-vaults JSON field, currently serialised seconds for
  timedelta values.
- `lockup_days`: explicit top-vaults JSON field in days.
- `VaultMetadata.lockup_days`: days.
- `pair.other_data["vault_lockup_days"]`: days.

Do not change generic timedelta serialisation in `export_lifetime_row()` because
it may affect unrelated fields and downstream consumers.

## Implementation steps

### 1. Patch trade-executor defensively

Add a small helper in trade-executor to normalise vault lockup metadata to days
before the value is displayed or used for expiry calculations.

Candidate location:

- `tradeexecutor/strategy/dex_data_translation.py`

Candidate helper:

```python
def normalise_vault_lockup_days(value: object) -> float | None:
    ...
```

Behaviour:

- Return `None` for `None`, non-numeric, non-finite, or non-positive values.
- Return `None` for `0.0`; in trade-ui this means "no lockup to display".
- Handle `datetime.timedelta` and `pd.Timedelta` defensively by converting
  them to `total_seconds() / 86400`.
- Treat normal values as days.
- Treat implausibly large day values as legacy seconds and divide by `86400`.

Suggested threshold:

- If value is greater than `3650`, treat it as seconds.
- Use a named constant, e.g. `LEGACY_LOCKUP_SECONDS_THRESHOLD_DAYS = 3650`,
  with a comment explaining that values above 10 years are not meaningful for
  trade-ui and are treated as legacy seconds-valued metadata.

Reasoning:

- A real ERC-4626 vault lockup above 10 years is not useful for trade-ui.
- Current broken Plutus value is `2592000.0`, so the conversion is unambiguous.
- This avoids showing broken metadata while upstream fixes propagate.

Use the helper when copying metadata:

```python
pair.other_data["vault_lockup_days"] = normalise_vault_lockup_days(metadata.lockup_days)
```

This is enough to fix both trade-ui static fallback and future stored
`vault_lockup_expires_at` values, because both read the pair-level field.

### 2. Add trade-executor regression tests

Update focused tests:

- `tests/test_vault_metadata_translation.py`
  - Keep the existing `30.0 -> 30.0` test.
  - Add `2592000.0 -> 30.0` to cover legacy seconds-valued metadata.
  - Add an idempotency assertion:
    `normalise_vault_lockup_days(normalise_vault_lockup_days(2592000.0)) == 30.0`.
  - Pin the threshold boundary:
    `3650.0 -> 3650.0` and `3651.0 -> 3651.0 / 86400`.
  - Cover `datetime.timedelta(days=30)` and `pd.Timedelta(days=30)` if the
    helper supports them.
  - Cover a large `datetime.timedelta` input so unambiguous timedelta values do
    not get divided again by the legacy numeric threshold.
  - Cover `0.0 -> None`.
- `tests/cli/test_trade_ui_settlement_eta.py`
  - Add a case where a pair carrying `vault_lockup_days = 30.0` renders
    `~30.0d`.
  - If a direct helper test is enough, avoid duplicating UI coverage.
- `tests/test_vault_lockup_expiry.py`
  - Existing 30 day expiry coverage is sufficient because the fix normalises
    `vault_lockup_days` at pair translation time, before state code reads it.

Run only targeted tests:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/test_vault_metadata_translation.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/cli/test_trade_ui_settlement_eta.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/test_vault_lockup_expiry.py
```

Use the parent repo Poetry environment if this branch is in a worktree whose
editable install points at the parent checkout.

### 3. Patch trading-strategy canonical loader

In the trading-strategy repository, update
`tradingstrategy/alternative_data/vault.py`.

Preferred parsing:

```python
lockup_days = entry.get("lockup_days")
if lockup_days is None:
    lockup_seconds = entry.get("lockup")
    lockup_days = lockup_seconds / 86400 if lockup_seconds is not None else None
```

Add a helper if useful, for example `_parse_lockup_days(entry: dict)`.

Test:

- A top-vaults JSON entry with `lockup: 2592000.0` loads as
  `VaultMetadata.lockup_days == 30.0`.
- A top-vaults JSON entry with `lockup_days: 30.0` loads as
  `VaultMetadata.lockup_days == 30.0`.
- If both are present, `lockup_days` takes priority.

### 4. Patch eth_defi export contract

In the eth_defi repository, update the top-vaults JSON export path so each
exported row includes an explicit `lockup_days` field.

Likely location:

- `eth_defi/research/vault_metrics.py`
- `export_lifetime_row()`

Because `export_lifetime_row()` is generic, prefer adding `lockup_days` before
or during row export instead of changing all timedelta serialisation.

Expected output for a 30 day lockup:

```json
{
  "lockup": 2592000.0,
  "lockup_days": 30.0
}
```

Keep `lockup` for backwards compatibility until all consumers can rely on
`lockup_days`.

Test:

- A row with `lockup = datetime.timedelta(days=30)` exports both
  `lockup == 2592000.0` and `lockup_days == 30.0`.
- A row with missing lockup exports `lockup_days is None`.

### 5. Release and data refresh order

Recommended order:

1. Merge trade-executor defensive fix.
2. Merge trading-strategy loader fix.
3. Merge eth_defi explicit export fix.
4. Refresh top-vaults metadata.
5. Remove or relax trade-executor defensive conversion only if the legacy data
   contract is retired.

The trade-executor defensive patch can ship first and immediately fixes the
operator-facing UI as long as the current metadata value is present.

## Risks and edge cases

- Some existing data may already contain true days. The normalisation helper
  must leave small values like `30.0` unchanged.
- A lockup in seconds below the threshold would not be converted. This is
  acceptable for known affected vaults because Plutus 30 days is `2592000.0`.
- A real lockup longer than 10 years would be converted incorrectly. Such a
  value is not meaningful for the current trade-ui workflow and should be
  treated as invalid metadata.
- The helper must be idempotent. Once trading-strategy and eth_defi emit
  explicit day values, the trade-executor defensive conversion must not divide
  the value again.
- Do not overwrite HyperCore live per-user lockup timestamps. This fix only
  normalises generic metadata durations.
- Existing persisted positions with already broken year-9123
  `vault_lockup_expires_at` are not fixed by normalising pair metadata. Treat
  this as a separate operator repair step: detect estimated lockups where
  `vault_lockup_estimated is True`, the expiry is implausibly far in the
  future, and the pair has normalisable lockup metadata, then rewrite the expiry
  from the original deposit timestamp plus the normalised lockup days. Do not
  include this repair in the first code patch unless a live state file needs it.

## Acceptance criteria

- Plutus Hedge lockup displays as about `~30.0d`, not `~2592000.0d`.
- New deposits derive a 30 day estimated expiry from Plutus metadata.
- Existing tests for `30.0` day metadata continue to pass.
- Legacy seconds-valued JSON metadata is covered by tests.
- The defensive normalisation helper is idempotent and has pinned threshold
  behaviour.
- The plan for upstream repositories preserves backwards compatibility.
