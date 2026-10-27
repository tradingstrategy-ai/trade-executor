# Vault closure and strategy interaction

Use `tradeexecutor.strategy.vault_closure` for both strategy decisions and charts. A closed **deposit or redemption operation** is not necessarily a permanently shut-down vault. Do not infer closure directly from an ERC-4626 maximum, a missing sample, or a trading-policy restriction.

```python
from tradeexecutor.strategy.vault_closure import (
    VaultClosureOutcome, get_vault_closure,
    is_vault_closed, can_interact_with_vault,
)

assessment = get_vault_closure(
    pair, timestamp=timestamp, pricing_model=pricing_model, operation="deposit",
)
closed = is_vault_closed(assessment)
allowed = can_interact_with_vault(assessment)
if assessment.status is VaultClosureOutcome.unknown:
    # Report missing evidence separately from whether the strategy allows it.
    pass
```

## Outcomes and controls

| Enum outcome | Meaning | Default interaction |
| --- | --- | --- |
| `open` | Available evidence permits the operation | Allowed |
| `closed` | Explicit operation closure, meaningful zero capacity or a configured closed window | Refused |
| `unknown` | Missing, stale or unsuitable admission evidence | Protocol/history dependent |
| `policy_blocked` | Optional trading policy, rather than observed venue closure | Refused |
| `account_restricted` | Live owner lockup or owner-scoped zero capacity | Refused; policy overrides cannot waive this |

`is_vault_closed()` returns true only for `closed`. `can_interact_with_vault()` returns the assessment's interaction permission. Both delegate to `get_vault_closure()`; keyword arguments are forwarded and misspellings raise an error. The latter returns the frozen `VaultClosure` dataclass with enum `status`, `blocked`, `source`, `reason`, `message` and `max_amount`.

Keyword-only controls are `timestamp`, `operation`, `pricing_model`, `historical_state`, `window_override`, `deposit_check`, `position`, `include_strategy_policy`, `unknown_is_closed`, `trust_protocol_state` and `apply_hypercore_cutoff`. See the [API reference](api/vault-closure.md) and function docstrings for their contracts. Changes to policy belong in explicit call-site options, never ad hoc chart filters. Pass a captured assessment directly to either boolean helper to avoid a second live request. Captured assessments accept no query kwargs; requesting a different vault, operation or timestamp requires a new query.

## Protocol cases and evidence

| Case | Interpretation and evidence |
| --- | --- |
| Explicit synthetic window | Takes precedence over recorded history. NB04/NB05 HYPE++ (Arbitrum `0x75288264fdfea8ce68e6d852696ab1ce2f3e5004`) uses an assumed schedule. A live `isFunding=false` on 27 October 2026 does not establish historical closure. [PR 1570](https://github.com/tradingstrategy-ai/trade-executor/pull/1570), [PR 1612](https://github.com/tradingstrategy-ai/trade-executor/pull/1612). |
| Lagoon | ERC-7540 `maxDeposit` describes settled claims, not acceptance of new requests. Syntropia Core `0x1b2cb79a4564206f53ba80b4d780f251b4ae6765` and Boosted `0x8df3deba711ae4a9af16cbca5e4fbb1402f036d5` had zero maxima while unpaused, permissionless and accepting requests on 27 October 2026. Historical generic fields are unknown admission; the existing backtest assumption allows deposits. Live checks use the request adapter. |
| Morpho V2 | ERC-4626 maxima deliberately return zero. Api3 dCOMP USDC `0x36cfe1568461e499391ef0a555300f1ae2da2439` and Alpha Forex `0x153bd1abe60104bd46aa05a27fa12d1346d64a57` had zero maxima and no external gates on 27 October 2026. A transaction was not simulated. Generic maxima do not establish closure. |
| Yearn V3 | A zero-address query may return zero while a nonzero receiver can deposit. Chimi USDC (Base `0xcb2f26898c0893c0bdd5cf76417cf9b2258af0ed`) and Turbo (`0x0c352ab624313ae28ed4073dc5d469eb36e164c6`) demonstrated this on 27 October 2026. Generic historical fields are not owner-aware admission evidence. |
| Redemption maxima for those three groups | Claimable requests, always-zero protocol maxima or a zero-address share balance do not prove a vault-wide withdrawal closure. Ignore these generic caps by default, but honour an explicit redemption permission flag. Live account/position checks remain authoritative. |
| Native Hyperliquid permission | After the collection cutoff, explicit permanent closure or leader-disabled deposits blocks deposits. The pre-cutoff unknown/allowed assumption takes precedence over historical flags; it does not assert those flags were open. A historical native deposit query requires a timestamp. HLP parent permission exceptions are handled by the shared eth-defi protocol classifier. Before the collection cutoff, admission is an explicit unknown/allowed assumption; afterwards missing or stale permission is unknown/refused. [PR 1660](https://github.com/tradingstrategy-ai/trade-executor/pull/1660), [PR 1670](https://github.com/tradingstrategy-ai/trade-executor/pull/1670). |
| Hyperliquid leader share | BitRocket, HYPErQuantum4 and Cold Process were API-open on 27 October 2026 with leader shares near 5%. The 5.5% safety threshold is trading policy, not a proven follower-deposit prohibition. A zero policy cap remains enforced but is counted as `policy_blocked`. [PR 1671](https://github.com/tradingstrategy-ai/trade-executor/pull/1671). |
| Other historical vaults | Explicit false flags or meaningful zero capacities restrict their respective operation. Missing data stays unknown. Do not ignore every zero: HYPE++, Earn ctUSD and Blackhole have meaningful restrictions. [PR 1542](https://github.com/tradingstrategy-ai/trade-executor/pull/1542). |
| Live account restrictions | Owner availability, account lockups and request permissions remain with live adapters; they are not reconstructed from historical prices. Unknown/fetch failures and policy blocks must not be described as permanent shutdown. |
| Universe exclusions | Whitelisted/data-only vaults are not made investable by an admission check. Keep selection rules separate: [getting-started PR 40](https://github.com/tradingstrategy-ai/getting-started/pull/40), [PR 52](https://github.com/tradingstrategy-ai/getting-started/pull/52), [PR 56](https://github.com/tradingstrategy-ai/getting-started/pull/56). The separate [LLP selection PR #1687](https://github.com/tradingstrategy-ai/trade-executor/pull/1687) excludes Lighter LLP by default because it requires staked LIT; this closure module does not itself change universe membership. |

The Lagoon/Morpho/Yearn interpretation follows [the earlier Hyper AI investigation](https://github.com/tradingstrategy-ai/trade-executor/pull/1622#issuecomment-5236615753), which examined observations dated 5 August 2026. Protocol definitions: [ERC-7540 request lifecycle](https://eips.ethereum.org/EIPS/eip-7540#request-lifecycle), [Morpho V2 ERC-4626 compliance](https://github.com/morpho-org/vault-v2#erc-4626-compliance). Current live observations corroborate protocol interpretation; they do not backfill historical permission.

## Ownership and integration

- **eth-defi:** one Hyperliquid protocol classifier, shared by exporters, live pricing and execution revalidation.
- **trading-strategy:** loads observations, calls that classifier when projecting legacy sidecars, preserves explicit null capacities and selects coherent original-clock evidence. It does not import the executor or implement another leader-share threshold.
- **BacktestPricing:** performs as-of/freshness selection, then supplies evidence to `get_vault_closure()`; existing structured pricing methods are adapters around that rule.
- **GenericPricing:** routes to the appropriate provider without reinterpreting fields.
- **AlphaModel / PhaseAwareAlphaModel:** use `can_interact_with_vault()`. Only explicit window-gated vaults may park deposits; closure checks do not change ranking or membership.
- **Vault window analytics:** count shared enum outcomes and expose policy blocks and unknowns separately. Coverage/age/size lines are not counts of immediately investable vaults.

`strategy.deposit_check` remains a compatibility import shim. New code imports `strategy.vault_closure`. Amount-aware sizing, funding, transaction simulation and execution-time checks remain separate responsibilities. `trust_protocol_state=True` is only for independently verified protocol-specific admission evidence, not the generic zero-address fields.

## Captured checks and live evidence

`DepositCheckResult` retains the shared enum outcome and source when created by the historical rule, including through JSON serialisation. Older adapters without these optional fields are classified from their recorded reason and API evidence. A missing owner or unknown limit is not positive evidence that deposits are open. Zero live owner capacity and withdrawal lockups are `account_restricted`, not global vault closures; `include_strategy_policy=False` never bypasses them. Positive amount limits still require separate amount-aware sizing.

A captured `deposit_check` accepts policy controls only, not a replacement timestamp, position or historical interpretation. Directly passing a `VaultClosure` to either boolean helper accepts no query kwargs. Historical `apply_hypercore_cutoff=False` exists for legacy field-classification wrappers; normal strategy decisions retain the cutoff.
