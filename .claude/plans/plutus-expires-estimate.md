# Estimated vault lockup display plan

## Goal

Show an estimated lockup countdown in `trade-ui` after a successful deposit into PlutusDAO, D2 Finance, and any other vault protocol whose dataset metadata already exposes an estimated lockup duration.

Today `trade-ui` can show:

- Async deposit settlement ETA from `trade.other_data["vault_settlement_estimated_at"]` for Ostium / Lagoon-style queued deposits.
- Per-user HyperCore lockup expiry from `position.other_data["vault_lockup_expires_at"]`.

It cannot show the generic ERC-4626 estimated lockup duration exported by the vault metadata pipeline, so PlutusDAO and D2 positions fall back to `-` unless some protocol-specific runtime code writes `vault_lockup_expires_at`.

## Current findings

- `tradingstrategy.vault.VaultMetadata` already has `lockup_days: float | None`.
- eth-defi ERC-4626 scanning calls `vault.get_estimated_lock_up()` and stores it as vault metadata.
- PlutusDAO implements `get_estimated_lock_up()` as a 30-day modelling estimate.
- D2 Finance implements `get_estimated_lock_up()` from the current epoch length.
- `tradeexecutor.strategy.dex_data_translation.translate_trading_pair()` copies many `VaultMetadata` fields into `pair.other_data`, but not `lockup_days`.
- `tradeexecutor.cli.trade_ui_tui._format_lockup()` only renders pending async settlement and stored `position.other_data["vault_lockup_expires_at"]`.
- The HyperCore path is richer because valuation fetches a live per-user expiry and writes `vault_lockup_expires_at`.

## Design

Use a two-tier display model:

1. Prefer a concrete per-position expiry timestamp when available.
2. Fall back to an estimated protocol lockup duration from pair metadata.

Keep the UI behaviour state-driven. `trade-ui` should not make RPC calls or instantiate vault contracts.

### Data keys

Use these state keys:

- `pair.other_data["vault_lockup_days"]`: estimated lockup duration in days from `VaultMetadata.lockup_days`.
- `position.other_data["vault_lockup_expires_at"]`: concrete or estimated naive UTC ISO timestamp for this position.
- `position.other_data["vault_lockup_estimated"]`: optional boolean marker when the expiry was derived from `vault_lockup_days` rather than live protocol state.

Do not overload `vault_settlement_estimated_at`; settlement queue ETA and post-deposit redemption lockup are different concepts.

`vault_lockup_estimated` is not only informational. It controls:

- Whether a later deposit may extend an existing estimated expiry.
- Whether a later live protocol reader may replace an estimated expiry.
- Whether `trade-ui` prefixes the countdown with `~`.

## Implementation steps

### 1. Export metadata lockup days

Update `tradeexecutor/strategy/dex_data_translation.py`:

- In the `VaultMetadata()` branch, copy `metadata.lockup_days` into `pair.other_data["vault_lockup_days"]`.
- Preserve `None` as `None`; do not invent a value when metadata is missing.
- Add or update a metadata translation test proving `lockup_days` survives pair translation.

This fixes the data path for PlutusDAO, D2, and any other vault where the data pipeline has populated `VaultMetadata.lockup_days`.

### 2. Persist estimated per-position expiry after deposits

Add a helper near vault execution/state code, for example:

```python
def maybe_set_vault_lockup_expiry(position: TradingPosition, trade: TradeExecution) -> None:
    ...
```

Behaviour:

- Only act for successful buy/deposit trades on vault pairs.
- Read `lockup_days` from `trade.pair.other_data["vault_lockup_days"]`.
- If `lockup_days` is `None` or `<= 0`, leave the position unchanged.
- Use the best available deposit timestamp:
  - `trade.executed_at` when present.
  - Otherwise `trade.opened_at` as a fallback.
- Derive `new_estimated_expiry = deposit_timestamp + datetime.timedelta(days=lockup_days)`.
- If no existing `vault_lockup_expires_at` exists, store `new_estimated_expiry` as a naive UTC ISO string and set `vault_lockup_estimated = True`.
- If an existing expiry exists and `vault_lockup_estimated` is true, store `max(existing_expiry, new_estimated_expiry)` so top-up deposits can extend the displayed lockup.
- If an existing expiry exists and `vault_lockup_estimated` is false or missing, treat it as a live/concrete protocol value and do not overwrite it from metadata.

Wire this helper where successful vault buys are marked complete:

- Synchronous ERC-4626 deposits: after the deposit trade is marked successful and the position exists.
- Async ERC-4626 deposits: after the claim succeeds and the trade becomes successful, not when the request first enters `vault_settlement_pending`.
- Backtest settlement: after async settlement success, so backtests and live state match.

Do not apply this to sells/redemptions.

### 3. Add UI fallback for metadata-only positions

Update `tradeexecutor/cli/trade_ui_tui.py`:

- Keep the current priority:
  1. Pending async settlement.
  2. `position.other_data["vault_lockup_expires_at"]`.
  3. Generic warning flags.
- If no `vault_lockup_expires_at` exists but the pair has `vault_lockup_days`, show a compact estimated duration, e.g. `~30.0d`.
- This fallback is useful before the first valuation/execution cycle has persisted a position expiry.
- Use `vault_lockup_expires_at` once available, because it decays correctly as time passes.
- If `position.other_data["vault_lockup_estimated"]` is true, prefix the countdown with `~`, e.g. `~29.4d`.
- If `vault_lockup_estimated` is false or missing, render the countdown without `~` because it came from a concrete protocol reader.
- The metadata-only `~30.0d` fallback is static and does not decay. It should only be treated as a pre-position / pre-persistence estimate.

Consider a slightly wider Lockup column if strings like `eligible 18.0h` and `~30.0d` need more room. The current width is 14 and likely sufficient.

### 4. Handle D2 epoch data carefully

D2’s `lockup_days` can be time-sensitive because it is derived from the current epoch.

For the first implementation:

- Trust the dataset-provided `lockup_days`.
- Prefer concrete `redemption_next_open` metadata when present and later than the deposit timestamp, if it is already carried in `pair.other_data`.

Follow-up improvement:

- Add an optional live ERC-4626 lockup reader for protocols where a concrete unlock timestamp is cheaply available, similar in shape to the HyperCore `lockup_func`.
- D2 can use `fetch_redemption_next_open()` / current epoch end.
- PlutusDAO likely cannot provide a deterministic per-user expiry because openings are admin-controlled, so it should stay estimated.
- Add follow-up tests proving a later live protocol reader can replace an estimated `vault_lockup_expires_at` and clear/set `vault_lockup_estimated = False`.

### 5. Tests

Add focused tests only:

- Metadata translation:
  - `VaultMetadata(lockup_days=30.0)` becomes `pair.other_data["vault_lockup_days"] == 30.0`.
- trade-ui rendering:
  - A vault position with `vault_lockup_expires_at` still shows the countdown.
  - A vault position with only `pair.other_data["vault_lockup_days"]` shows `~30.0d`.
  - Pending async settlement still takes priority over metadata lockup.
  - Vault display flags remain the final fallback.
- State/execution helper:
  - Successful vault buy stores estimated expiry from `executed_at + lockup_days`.
  - Existing concrete `vault_lockup_expires_at` is not overwritten.
  - Existing estimated `vault_lockup_expires_at` is extended with `max(existing, new)`.
  - Sell trades do not update lockup.
- trade-ui estimated marker:
  - Estimated `vault_lockup_expires_at` displays with `~`.
  - Concrete `vault_lockup_expires_at` displays without `~`.

Run only targeted tests:

```shell
source .local-test.env && poetry run pytest tests/test_vault_metadata_translation.py
source .local-test.env && poetry run pytest tests/cli/test_trade_ui_settlement_eta.py
```

Add the new execution-helper test to the narrowest existing module or a new small unit module, then run that specific test file.

## Risks and edge cases

- An estimated lockup is not the same as a hard protocol guarantee. Mark display-only fallback values with `~` and keep exact timestamp values unprefixed.
- An estimated `vault_lockup_expires_at` must also render with `~`; otherwise metadata-derived timestamps look exact.
- Position expiry based on `position.opened_at` can be wrong for later top-up deposits. Use the successful buy trade timestamp instead.
- Multiple top-up deposits can extend or overlap lockups. Conservative behaviour is to set the expiry to `max(existing_expiry, new_estimated_expiry)` when the existing expiry is also estimated.
- Do not overwrite HyperCore’s live per-user expiry with generic metadata.
- D2 epoch timing may change between dataset refreshes. A future live reader should improve this, but the metadata fallback is still better than showing no lockup information.
- Backtests need the same state keys as live execution so exported state and trade-ui are consistent.

## Acceptance criteria

- PlutusDAO and D2 vault pairs loaded from `VaultMetadata` carry `vault_lockup_days`.
- After a successful deposit, vault positions can expose `vault_lockup_expires_at` without a protocol-specific live reader.
- `trade-ui` shows an estimated lockup duration/countdown instead of `-` for metadata-backed vault lockups.
- Existing Ostium/Lagoon pending settlement display is unchanged.
- Existing HyperCore live lockup display is unchanged.
