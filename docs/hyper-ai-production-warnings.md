# Hyper-AI production warnings

Review of the v1472 startup on 26 October 2026. These changes affect diagnostic
severity only; they do not change selection, sizing, execution or accounting.

| Warning | Finding and treatment |
| --- | --- |
| Missing realised profit for #18 and #19 | Zero-quantity repair placeholders, with no acquired assets or balance updates. Suppress the warning for empty administrative closures. |
| Missing realised profit for #532 | The 28 September opening failed before vault settlement; no executed quantity, reserve allocation or unresolved capital remains. Quiet the expected missing P&L, while retaining warnings for funded, started, broadcast or unresolved positions. |
| Charts unavailable for 528 pairs | The legacy technical-image renderer supports at most five pairs. Log its expected limit at INFO in both refresh paths; keep genuine drawing exceptions at WARNING. This does not add chart support. |
| Stale history for 82 vaults | Histories can be retained for research after an unheld vault falls below the strategy TVL floor. Summarise those at INFO only when current holdings and a floor are known and both metadata and latest historical TVL are below the floor. Historical TVL matters because strategy indicators may forward-fill it. Held, eligible and unknown-TVL histories still warn. Refresh holdings for every live universe construction, including frozen and pending positions. |
| dRPC startup verification failed | An actual provider availability failure, with two other providers available. Keep the warning. Check the host's authenticated RPC configuration and provider availability separately; do not print credentials. |
| Remote history HEAD returned 504 | The remote metadata request failed, even though the verified local history was fresh. Preserve the availability warning; daily candle flooring itself is not an error. |
| Chain 9993 metadata cannot parse | A record in the global dataset is unsupported by the installed `tradingstrategy.chain.ChainId`. Keep the compatibility warning; resolve chain semantics in the dataset producer and trading-strategy client rather than fabricating a chain mapping in the executor. |

## Previous work

An unfinished local branch, `fix/hyper-ai-actionable-warnings`, already contained
reporting and stale-history diagnostic changes but had no PR. This branch carries
that work forward on current master, leaving the original working tree intact.

Related merged PRs are [#1417](https://github.com/tradingstrategy-ai/trade-executor/pull/1417),
[#1418](https://github.com/tradingstrategy-ai/trade-executor/pull/1418) and
[#1432](https://github.com/tradingstrategy-ai/trade-executor/pull/1432) for vault
freshness/cutoff diagnostics, [#1441](https://github.com/tradingstrategy-ai/trade-executor/pull/1441)
for valuation scheduling diagnostics, and
[#1635](https://github.com/tradingstrategy-ai/trade-executor/pull/1635) for historical
profitability. Their fixes are already merged; none makes empty administrative
closures require a realised-profit warning.

The dependency's [RPC fallback PR #1127](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1127)
intentionally retains warnings when a configured provider fails verification.
The client's [metadata PR #258](https://github.com/tradingstrategy-ai/trading-strategy/pull/258)
intentionally reports malformed/unsupported records and propagates environment
failures. These warnings must not be hidden by a blanket logging-level change.

No overlapping open PR was found among the currently open executor PRs. Runtime
state and production configuration are not modified by this work.
