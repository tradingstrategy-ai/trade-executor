# Vault closure API

Module: [`tradeexecutor.strategy.vault_closure`](../../tradeexecutor/strategy/vault_closure.py).

| Public API | Result |
| --- | --- |
| `get_vault_closure(vault, *, ...)` | `VaultClosure` assessment |
| `is_vault_closed(vault, **kwargs)` | `bool`: outcome is `VaultClosureOutcome.closed` |
| `can_interact_with_vault(vault, **kwargs)` | `bool`: chosen policy allows the operation |
| `VaultClosureOutcome` | Enum: `open`, `closed`, `unknown`, `policy_blocked` |
| `VaultClosure` | Frozen evidence and decision record; `as_deposit_check()` adapts it to existing diagnostics |

All three functions accept the following keyword controls through `get_vault_closure`:

| Keyword | Default | Contract |
| --- | --- | --- |
| `timestamp` | `None` | Naive UTC decision time; provide it for historical decisions |
| `operation` | `"deposit"` | `"deposit"` or `"redemption"` |
| `pricing_model` | `None` | Preferred provider route; preserves historical clocks and live owner context |
| `historical_state` | `None` | Mapping already selected as-of and checked for freshness by the caller |
| `window_override` | `None` | Explicit synthetic schedule; takes precedence over historical fields |
| `assessment` | `None` | Reuse a previous assessment without requesting new evidence |
| `deposit_check` | `None` | Adapt an existing structured deposit check; deposit operation only |
| `position` | `None` | Live redemption position/lockup context |
| `include_strategy_policy` | `True` | Enforce known policy blocks; disabling does not change the evidence outcome |
| `unknown_is_closed` | `None` | Override whether unknown evidence blocks interaction; enum remains `unknown` |
| `trust_protocol_state` | `False` | Opt into separately verified protocol-specific historical admission data |
| `apply_hypercore_cutoff` | `True` | Preserve the historical pre-collection assumption and later unknown/refused default |

Input routes are mutually exclusive. Reusing an assessment cannot change its policy options. Unsupported keywords/operations and timezone-aware historical timestamps fail explicitly. The cutoff switch is intended for scoped legacy field classification, not ordinary strategy calls.

See [protocol cases, evidence and architecture](../vault-closure.md), [implementation docstrings](../../tradeexecutor/strategy/vault_closure.py), and [historical analytics](../../tradeexecutor/analysis/vault_windows.py).
