# Generic ERC-4626 support through perform-test-trade

## Objective and boundaries

Subplan of [NB05 live support](nb05-erc4626-live-support.md), revised on
2026-10-29 after Claude Opus 5.5's implementation-scope review.

Make the public `perform-test-trade` command reliably open and close a small
synchronous ERC-4626 position through ordinary `TradeExecution`, pricing, routing
and Lagoon Safe/Guard custody. Cover generic ERC-4626, Morpho V1/V2, Yearn V3
and Yearn Compounders using their existing selected managers. Existing specialised
and asynchronous command routes must keep working.
The new public `--close-only` mode is limited to home-chain synchronous vaults;
it rejects Lighter accounts and satellite routes before treasury sync.

The assessment's 205 primary identities and 41 additional synchronous candidates
remain candidates, not certified deployments. Per-identity certification and the
347-row report belong to the parent assessment. This implementation must not
claim those vaults are supported merely because representative tests pass.

## Source baseline and existing support

Start from executor master `eb43974659488d70de8736ea4f8e2bf95b7a54a8`, which pins
eth-defi `d19a5a6b9fcd29914b8749fcb438cf068002850d`. The current branch has been
rebased onto that master. Generic guarded routing, protocol-selected managers,
receipt analysis and pre-broadcast state saving
already exist. The public command omits the persistence callback; its existing
helper accepts close-only but still quotes a buy and requires reserve cash.

## Necessary changes

| Repo | Change description | Before | After |
| --- | --- | --- | --- |
| trade-executor | Resolve chain/address pairs and correct help | Address lookup ignores the supplied chain | Require exactly one matching chain/address in the strategy universe |
| trade-executor | Persist live execution | Public command saves at completion | Install the existing pre-broadcast callback, pass the same callback through each helper branch and persist each completed/error leg. Simulation never writes the source state file |
| trade-executor | Expose single-pair `--close-only` | The helper accepts it but the command does not | Reject conflicting/bulk selections, skip buy quotes and reserve-cash requirements, require exactly one wholly test-owned position and retain existing pending-operation guards |
| trade-executor | Protect existing holdings | A test can reuse an ordinary position | Refuse non-test positions and unexplained balance updates; refuse new execution while a matching trade is unresolved |
| trade-executor | Check admission and assets | Test commands need refusal before creating a trade | Check the funded quote asset and call the selected manager's existing admission checks. Reject zero deposit quotes. Keep capacity interpretation shared with strategy execution |
| trade-executor | Verify full closure | A successful sell can leave residual shares | Require a closed test position and exhausted test-owned shares before declaring a round trip complete. Refuse insufficient redemption capacity; leave the position open and report it |
| eth-defi | Preserve exact requested shares in shared managers | Legacy percentage corrections can change calldata without changing request metadata | Disable implicit resizing in manager request construction. Morpho V1 and Yearn recognise only bounded rounding and simulate that exact redemption. Normal strategy and test trades use the same managers |
| trade-executor | Refresh terminal Lagoon NAV | Small realised losses can fall below the ordinary empty-queue NAV threshold | Add explicit `lagoon-settle --force-valuation` before the final investor withdrawal; retain Guard budgets, cooldowns and all other settlement checks |
| trade-executor | Add the manual tutorial | The Lighter tutorial tests a different product | Deploy Lagoon on Arbitrum, subscribe, use public buy-only/close-only calls for each representative, reconcile, then redeem the investor |

Follow [state persistence](../state-persistence.md): use existing state, transaction
records and callbacks. No separate progress markers or recovery journal.
Use one state writer during manual commands. Shares belong to the Safe; gas is
paid by the asset-manager EOA. Resolve each on its actual chain. A successful but
unparseable or zero-output receipt stays unresolved with its identity preserved;
never refund or repeat a possibly executed deposit. No new transaction-cancellation
command, partial-exit framework, zero-loss ledger representation, result schema,
manager-policy framework or production state migration is needed for this task.

## Focused verification

1. Exercise public Typer selection, callback wiring and close-only through happy
   and refusal paths, including an ordinary position and unresolved execution.
2. Exercise guarded Lagoon buy-only and a separate close-only invocation on pinned
   forks for each selected implementation. Record actual shares, net cash and
   receipts. Use known historical deployments and disclose funding/time changes.
3. Interrupt after transaction acceptance/confirmation, reload the existing state
   and refuse new execution until the trade is reconciled through the existing
   operator workflow. Check simulation leaves the original state byte-for-byte unchanged.
4. Run the relevant existing generic and shared-command regressions; retain
   satellite regressions where shared callback wiring affects those callers.
   Add tests only for changed behaviour; reuse existing fixtures and helpers.
5. If tests reproduce a manager sizing bug, add the smallest upstream fix and
   regression test. Do not update the dependency to unrelated upstream changes.

A non-zero exit with saved state is required for an unavailable close, residual
shares, a pending operation or execution/analysis failure. Completed buy-only is
intentionally successful. Logs and the ordinary state/trade history are the result;
there is no second diagnostic state machine.

## Manual integration tutorial: Arbitrum Lagoon vault

Deliver `scripts/lagoon/manual-erc4626-cli.sh` following
[`manual-lighter-subaccounts-cli.sh`](https://github.com/tradingstrategy-ai/trade-executor/blob/eb43974659488d70de8736ea4f8e2bf95b7a54a8/scripts/lagoon/manual-lighter-subaccounts-cli.sh)
from PR #1691: numbered explained steps, copyable public CLI commands, `run`
wrapper, heartbeat, traps, private run directory, sourceable `env.sh`, saved
state/deployment and ledger-driven per-leg resume. Never log keys or
credential-bearing RPC URLs. Store the representative list in the small test
strategy, not a separate manifest framework. Refuse changed strategy/configuration
on resume. Deployment and subscription are explicit operator actions using
`--deploy` and `--subscribe`; a rerun without them continues from existing state.
An interrupted deployment or subscription stops for operator reconciliation,
without tracking completion in separate marker files.

| Step | Action |
| --- | --- |
| 0 | Check required tools and Arbitrum RPC. Check investor funds before explicit deployment/subscription and use `check-wallet` before trading. Stop other state writers |
| 1 | Deploy a Lagoon vault with Arbitrum (42161) home and native USDC using `lagoon-deploy-vault`; whitelist the strategy's five vaults. Use the command's existing deployment artefact; if it is missing after an interruption, recover the deployment before continuing |
| 2 | Export Lagoon/Safe/module addresses from the deployment artefact, initialise a newly deployed vault once and check accounts/wallet. A missing state file for an existing deployment requires operator recovery |
| 3 | Use existing `lagoon-first-deposit` for the investor request, settlement and share claim. If interrupted, stop for operator reconciliation before resuming |
| 4 | For every required representative, call `perform-test-trade --pair '(arbitrum, 0xTARGET)' --amount N --buy-only --no-test-short`, then separately `perform-test-trade --pair '(arbitrum, 0xTARGET)' --close-only --no-test-short`. Refresh actual owner admission, verify receipts and check balances after each leg |
| 5 | Run `check-accounts`, `check-wallet`, `show-valuation` and `lagoon-settle --force-valuation` before the terminal withdrawal. Do not correct discrepancies away. Require all positions reconciled; honour Guard limits/cooldowns |
| 6 | Redeem the investor's Lagoon shares with `lagoon-redeem`; check actual returned USDC, residual holdings and realised PnL/fees. Never sweep while liabilities or pending operations remain |

Use the five Arbitrum representatives listed below, including the distinct
Compounder reader. An unavailable route stops with a non-zero status. Cross-chain
funding and asynchronous protocols belong to the parent plan.

Use the ordinary state and receipts to check a leg before retrying. An
already-completed buy continues to the corresponding close.
Ambiguous signed transactions stay unresolved. Keep amounts small; run one position
at a time, stopping on an unavailable exit. Preserve protocol holdings, investor
shares and valuation when funds cannot return; resume the exit before investor
redemption. Do not increase funding automatically or write off residuals.

Validate shell syntax and the declared universe, and rehearse the command sequence
on pinned forks. The funded manual run is a separate operator action. Fork funding and
clock advancement must be labelled; none proves
current mainnet liquidity. Writing this implementation does not execute a funded
manual deployment or certify every recorded vault.

## Review and simplifications

Claude CLI `claude-opus-5-5` completed a plan-text review on 2026-10-29 specifically
for overengineering and necessity. Its recommendations are reflected above:
reuse existing execution/receipt mechanisms; remove the new cancellation command,
broad accounting changes, policy/result frameworks and tutorial manifest system;
retain focused correctness fixes and the explicitly requested manual tutorial.
A dependency change is conditional on a reproduced bug, not a prerequisite.

The implementation code was then reviewed with the same Claude model. The
completed review reported no blocking findings at that revision.

Following the user's scope correction, remove the diagnostic-specific receipt
recovery and investor transaction journal. Reuse the executor's existing state
store/callback and `lagoon-first-deposit`. Unresolved trades stop for the existing
operator reconciliation workflow; `perform-test-trade` does not add a second
repair path. This simplification is covered by the focused tests below.

The subsequent overengineering review removed the diagnostic-specific redemption
branch, moved rounding handling into the shared eth-defi managers, removed shell
progress markers, and restricted this tutorial to Arbitrum.

## Implementation and evidence

Implemented against the baseline above with a focused change in the local
`deps/web3-ethereum-defi` checkout. Ship that upstream change and update the
executor dependency revision together; the original pin alone does not contain
the shared sizing fix. No unrelated upstream updates are required.
The tutorial strategy declares these Arbitrum representatives:

| Implementation | Vault | Evidence |
| --- | --- | --- |
| Generic ERC-4626 entry points (Euler deployment) | `0x05d28a86e057364f6ad1a88944297e58fc6160b3` | Guarded public buy/close, full shares exhausted |
| Morpho V1 | `0x4b6f1c9e5d470b97181786b26da0d0945a7cf027` | Exact simulated full exit; observed sub-asset-unit capacity rounding |
| Morpho V2 | `0x55a2b207b0074e13adcb858950a81b7a04775e0f` | Existing specialised manager and exact guarded redemption |
| Yearn V3 | `0x2b0b6376083c6e1f376c7439f328436a673f333c` | Exact simulated full exit; observed one-raw-share capacity rounding |
| Yearn Compounder | `0xcacc53bacce744ac7b5c1ec7eb7e3ab01330733b` | Selected Compounder reader, guarded public buy/close |

`tests/lagoon/test_erc4626_manual_cli_e2e.py` deploys Lagoon using the public
command on Arbitrum block **487039644**, subscribes the investor, opens/closes
all five representatives through separate public commands, checks accounts after
each, forces terminal NAV refresh and redeems the investor. It reconciles returned
USDC with actual protocol cash flows. Only investor USDC/gas are injected; no
protocol liquidity or operator settlement is injected. The immediate exit trial
uses `--anvil-time-skip-seconds 0`. The matrix is entirely on Arbitrum.

Verified on 2026-10-29:

- Two focused CLI unit tests: chain/address ambiguity, incompatible command
  modes, shared capacity refusals, zero-share quotes and shell syntax.
- Shared eth-defi manager tests: exact calldata and request quantities, bounded
  Morpho/Yearn rounding, insufficient capacity/shares and zero/reverting payouts.
- Existing eth-defi Yearn preflight tests and the generic IPOR redemption fork
  regression also pass with shared exact sizing.
- Three fork scenarios: complete tutorial using `lagoon-first-deposit`; interrupted
  all-cash deposit refuses further execution; ordinary-position refusal and
  failed-simulation state isolation.
- Sixteen existing shared regressions for satellite async control flow, guarded
  Lagoon settlement and vault receipt/reconciliation helpers.
- Existing all-pairs CLI regression; the Guard over-budget test also passes with
  forced valuation enabled, preserving its queued flows.

Tests use `.local-test.env`, the parent Poetry environment and worktree source
precedence. Run the focused tests with:

```shell
source .local-test.env
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" poetry run pytest deps/web3-ethereum-defi/tests/erc_4626/test_redemption_request_sizing.py
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" poetry run pytest tests/cli/test_perform_test_trade_erc4626.py
PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" poetry run pytest -n auto --dist loadgroup tests/lagoon/test_erc4626_manual_cli_e2e.py
```

Recovery remains outside this command. A missing/unconfirmed receipt, approval-only
execution, request-preparation error after execution starts, or a reverted/frozen
position stops for operator reconciliation; the command does not unfreeze, cancel
or re-sign it automatically. A known preflight capacity refusal before a new
trade leaves the position available for a later close attempt. Receipt-backed
zero-output losses remain unresolved rather than being automatically credited or
retried. The funded Arbitrum tutorial completed on 2026-10-29: all five declared
representatives opened and closed, and the investor redeemed 9.999997 USDC from
the 10 USDC subscription. See the [live run report](../reports/erc4626-manual-live-2026-10-29.md)
for transactions, interruptions and final balances. This evidence does not
certify all 205 primary identities or the full 347-vault assessment.

The shared managers retain their existing zero-limit interpretation: some adapters
use zero as unadvertised capacity. The diagnostic does not add its own class-based
policy. Protocol-specific capacity corrections belong in those shared managers.

## PR cleanup review

Cleanup on 2026-10-29 removed duplicate state writes, moved the public close-only
guard before treasury sync and native/satellite dispatch, and moved the unchanged
target list to a strategy constant. The tutorial helper now uses Typer with the
main CLI's traceback-local suppression. Lighter recovery guidance and the
strategy-wide effects of exact redemption sizing are documented.

Claude CLI `claude-opus-5-5` independently reviewed the implementation and the
corrections. Its completed final follow-up confirmed the findings were resolved
with no remaining blocking or high-confidence issue in those fixes. UK English,
sentence-case headings and local Markdown links were checked.

Fourteen focused tests passed during cleanup: two CLI/admission tests, three
pinned-fork scenarios and nine existing satellite regressions. The fork round
trip now explicitly verifies that normal settlement skips a small NAV change
with an empty queue, while forced settlement posts and settles it. Shell syntax,
the helper's public `pairs` command, a synthetic failure without traceback locals,
and both repositories' whitespace checks also passed. No additional funded
transaction was sent for cleanup.

The executor integration pins the shared sizing fixes published in
[eth-defi PR #1700](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1700).
Merge that dependency before the executor integration. The live run's frozen
strategy remains unchanged in its private run directory; later cleanup edits are
covered by the checks above.

CI exposed the same one-raw-share capacity rounding in EulerEarn. Its selected
manager now permits only that discrepancy after a positive exact simulation;
generic Euler capacity policy is unchanged. The existing guarded matrix verifies
exact request/event quantities and zero residual shares for all seven profiles.
The old Yearn dust expectation and the raw-share probe's missing capacity fixture
were corrected. All seven guarded profiles and eight focused unit cases passed
locally after reproducing the three original CI failures.
