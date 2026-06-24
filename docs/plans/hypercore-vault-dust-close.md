# Fix plan: Hypercore vault dust pollutes closed positions

## Problem

The hyper-ai strategy (Lagoon vault-of-vaults on Hyperliquid, chain 999) shows
hundreds of empty (zero-quantity) closed positions in its UI.

Analysis of the production state file (`hyper-ai`, 449 closed positions) shows:

- **403 / 449** closed positions have zero executed quantity.
- **376** of those are single synthetic `repair` trades with
  `executed_quantity == 0`, no blockchain transactions, `opened_at == closed_at`,
  and note `"Auto-closed as dust by correct-accounts (equity=$X)"` where X is
  ~$0.0001–$1.77 (median ~$0.63; 188 of them in the $1–2 band).
- The remaining 27 are `rebalance` trades that were planned but never filled
  (separate, minor issue — out of scope for this fix).

## Root cause

Lifecycle traced for one repeatedly-dusty vault (Super Moon,
`0xbe421ec5af0be9ea2b2a53b38a089ee352c9f325`):

1. Strategy buys the vault, then later exits (sell / redeem).
2. Hypercore withdrawals **cannot fully exit**: the protocol refuses an exact
   full withdrawal when NAV moves between planning and execution, so a
   ~1.5 USDC safety-margin residual is stranded on-chain.
3. The real position **is correctly marked closed at exit** —
   `TradingPosition.can_be_closed()` (`state.py:1043`) tolerates the residual
   because `HYPERLIQUID_VAULT_CLOSE_EPSILON = Decimal("2.00")`. This part
   already matches other trading pairs.
4. **Bug:** `create_missing_vault_positions()`
   (`tradeexecutor/strategy/account_correction.py`), called from
   `correct-accounts` on every cycle (~daily), scans every vault pair, finds the
   ~$1.50 of leftover on-chain shares with **no matching open position**, and
   manufactures a fresh zero-quantity `repair` position which it immediately
   closes as dust. Because the residual lives on-chain forever, every run
   re-detects the same dust and creates another empty closed position.

This is unique to the Hypercore path. For spot pairs, leftover sub-dust balances
are simply written off after close; account correction never manufactures
phantom closed positions for them.

## Fix

In `create_missing_vault_positions()`, when a vault has dust-level equity
(`equity < get_close_epsilon_for_pair(pair)`) and no open position, **write it
off and `continue`** — instead of creating + immediately closing a phantom
position. This mirrors how leftover spot dust is treated.

- The genuine exit already closes the real position, so nothing economic is lost.
- The residual is < $2 per vault and genuinely un-withdrawable.
- `_build_hypercore_vault_account_checks()` only checks open/frozen positions,
  and the docstring confirms Hypercore vaults "never enter the generic asset-map
  based correction path", so the written-off residual will not trigger a
  spurious mismatch elsewhere.

### Changed code

`tradeexecutor/strategy/account_correction.py` — dust branch of
`create_missing_vault_positions()`: replace the create-trade + mark_success +
mark_down + close_position block with a `logger.info(...)` + `continue`.

Above-dust equity still creates a real open position (unchanged).

## Tests

New: `tests/hyperliquid/test_correct_accounts_vault_dust_writeoff.py`

1. `test_vault_dust_is_written_off` — dust equity (< close epsilon), no open
   position → no trades created, no open/closed positions.
2. `test_vault_real_equity_still_opens_position` — equity above close epsilon,
   no open position → one open position created (guards the write-off does not
   suppress genuine missing positions).

Regression check: existing hypercore suites still pass
(`test_hypercore_accounting.py`, `test_close_dust_hypercore.py`,
`test_cli_repair_hypercore_dust_sample_state.py`,
`test_hypercore_account_checks_sample_state.py`) — 22 passed, 3 skipped.

## Review (codex CLI)

Verdict: **LGTM**, no downstream consumer relies on the old "create + close dust
position" behaviour; generic ERC-20 correction cannot see Hypercore equity and
`_build_hypercore_vault_account_checks()` only checks open/frozen positions; the
USD-equity vs epsilon comparison is sound. Two minor suggestions applied:

- Use `equity <= epsilon` (not `<`) so the threshold matches `can_be_closed()`,
  which treats an exact-epsilon residual as closeable dust.
- Added a repeat-call assertion to the dust test to lock the "no accumulation
  across cycles" regression explicitly.

## Out of scope / open questions

- The 27 unfilled-rebalance empty positions (planned buy never executed) are a
  separate issue.
- Lowering `HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW` would reduce stranded dust
  but cannot eliminate it (protocol refuses exact full exits). Not pursued here.
- Existing dust positions already in production state are not retro-cleaned by
  this change; `repair_hypercore_dust` / `correct-accounts` history remains.
