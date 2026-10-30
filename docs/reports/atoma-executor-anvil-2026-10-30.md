# Atoma executor Anvil validation

Validated on 2026-10-30 using real deployed Atoma contracts on an Arbitrum fork
at block **519150162**, with implementation
`0x9161be115867c347adfb4c88a80e759920de86dc`. Both Index and RWA completed the
executor lifecycle through a newly deployed Lagoon Safe and its real Guard.

The [integration plan](../plans/nb05-atoma-integration-subplan.md) defines scope.
The maintained mechanism is documented in the
[shared async vault guide](../../.claude/docs/vault-deposit-redeem.md).

## Exercised behaviour

The [black-box CLI tests](../../tests/erc_4626/test_atoma_async_cli.py) deploy via
`lagoon-deploy-vault`, subscribe 400 USDC, run `init` and repeated Typer `start`
commands with the [dummy strategy](../../strategies/test_only/atoma_async.py),
and reload the normal state file after commands. The strategy deposits 100 USDC
into each target, waits for unlocked shares, requests redemption once and leaves
settlement/claim completion to the existing resolver. No routing or receipt
results are mocked in the happy path.

Verified outcomes include pending escrow and burned-share account checks,
Guard refusal of direct redemption/operator calls, repeated pending starts,
attributed net claims and no duplicate trades. Final Safe cash/NAV is 399 USDC;
the investor recovers it through `lagoon-redeem`, ending with 499 USDC including
the 100 USDC retained before subscription. The Guard retains a 5,000 USDC gross
settlement budget over 24 hours.

| Target | Consumed shares | Net claim, USDC | Receipt withdrawal fee, USDC |
| --- | ---: | ---: | ---: |
| Atoma RWA | 92.043718995111 | 99.5 | 0.499999 |
| Atoma Index | 90.183127089591 | 99.5 | 0.5 |

The RWA gross claim rounds to 99.999999 USDC; receipt proceeds are already net.
Fees are retained as reporting evidence without another reserve debit.

Separate cases verify public `perform-test-trade --close-only` resume and locked
exit refusal, real post-settlement haircuts followed by zero payouts, and a crash
after actual claims pay but before receipt accounting. Restart recovers the
saved claim hashes before startup valuation/NAV and applies their cash once.
Another case persists the adapter's reconciliation-required marker and proves
that even a recent prior valuation cannot allow a NAV transaction.

## Spoofs and limits

- Funding transfers existing fork USDC from an impersonated holder and provides
  test gas. Protocol liquidity is not injected.
- Anvil time advances across the actual deposit lock and settlement boundaries.
  There is no verified cooldown-to-zero setter.
- The deployed operator is impersonated only to call real settlement methods.
  Separate loss snapshots impersonate operator/owner roles for real `pause()`
  and `resyncTotalAssets()` calls. The asset-manager Guard never gains those roles.
- The crash case deliberately interrupts receipt analysis after real payment.
  It does not return a fabricated receipt or successful accounting result.
- The simulation uses a shared local RPC with `SIMULATE=false` and a normal JSON
  state store. No production wallet or state is used.

These results establish mechanics at the pinned deployment state. Current live
admission, liquidity and operator availability were not certified. Aggregate
claims remain refused pending separately reviewed reconciliation; automatic
multi-trade payout allocation is outside this release.

## Local receipt evidence

One completed run used Safe `0x28E01E25c19a39aD183bB2276869B88F5E3E7063`:

| Target | Request hash | Claim hash |
| --- | --- | --- |
| RWA | `0x3c47d4e71b35e73c7540ae512aac611f1ea8d2a2cbdef43b9c456db990a5635c` | `0xd8e364925e757f760ba77db658e5739da691c3681da94c645edbd14f1932d374` |
| Index | `0x184663c6ce9affab7daf2028c66781bdcb76cc716604f9358eb522e23e7851fb` | `0x0e7bae36d5ac3e48dfd5db849667fe23eacc0e7c1e983f1fd747b8ca9a0fd86b` |

These are local fork receipts, not mainnet transactions. Each happy-path run
writes a diagnostic `atoma-fork-evidence.json` beside its temporary state file;
that report is never read for execution or recovery.

## Checks

Run focused cases with the repository's private environment loaded, the parent
Poetry environment selected, and worktree/dependency imports taking precedence:

```shell
source .local-test.env
export PYTHONPATH="$PWD:$PWD/deps/web3-ethereum-defi:$PWD/deps/trading-strategy:$PYTHONPATH"
timeout 360 poetry run pytest tests/erc_4626/test_atoma_async_cli.py -q
```

| Check | Result |
| --- | --- |
| Atoma guarded Typer CLI module, including both targets | 5 passed |
| Atoma partial/illiquid valuation and zero-payout state arithmetic | 2 passed |
| Shared vault settlement lifecycle | 17 passed |
| Public vault-test safety | 79 passed |
| Lagoon universe/deployment configuration | 2 passed |
| Existing state tests | 48 passed; 4 pre-existing internal-state fixture cases skipped |
| Existing Lagoon ERC-7540 real-fork lifecycle | 1 passed |
| Existing generic guarded ERC-4626 manual round trips | 1 passed (five target implementations) |
| Existing Atoma delayed-redemption backtest | 1 passed |
| Existing HyperCore accounting regressions | 21 passed |
| Upstream Atoma offline manager tests, including the consumer fixes | 46 passed |

No full suite, funded live execution or production Guard migration was run.
The dependency combines upstream `2939ee63` with the executor's original
`de5538c4` pin to retain its Euler fixes, plus the targeted request-reconstruction
and attributed-quote changes. Its package requirements are unchanged, so the
existing lockfile remains applicable. Dependency publication is separate from
this local validation.

## Review and delivery

Claude CLI with explicit `claude-opus-5-5` completed a repository-grounded review
and a focused follow-up. Its findings led to limiting specialised deployment
selection to Atoma, allowing receipt recovery when optional consumed-claim
calldata cannot be rebuilt, correcting tutorial wait outcomes and sharing the
early-recovery predicate. The follow-up reported no blocking or important
findings. Both reviews were read-only and did not run tests.

The focused checks above total **223 passed and 4 pre-existing skipped cases**.
During PR preparation, the executor was rebased onto `d6bf67aaf` and the
dependency updated to `40f13c744`. The repeated state, Atoma valuation and
guarded CLI checks passed with **55 passed and 4 pre-existing skips**; the
dependency's Atoma offline module also passed all **46 cases** again. Both
changelog conflicts were resolved by retaining the existing and new entries.
The changed shell wrapper passed `bash -n`; its Atoma universe selection was
checked with the read-only `pairs` command. The wrapper itself was not run
against mainnet. Local Markdown links and whitespace checks passed, and the
changed prose was checked for UK English and sentence-case headings.

The dependency tip for PR handoff is commit
`40f13c744` on `atoma-executor-consumer-fixes-20261030`. It contains the tested
consumer fix `29d6cb319`, the Euler-preserving merge `53a282040`, and current
upstream master. These dependency commits are published in
[web3-ethereum-defi PR #1703](https://github.com/tradingstrategy-ai/web3-ethereum-defi/pull/1703).
Merge that dependency fix before the executor change. The separate vault
inventory assessment was not present here; this report supplies evidence for
its two Atoma rows without changing an external PR.
