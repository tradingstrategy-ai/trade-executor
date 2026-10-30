---
name: sanity-check-plan
description: Check a software implementation plan against the conversation, repository documentation, existing code and common sense. Correct gaps, impractical validation and unnecessary complexity after writing or substantially revising a plan, or when asked to sanity-check one.
---

# Sanity-check plan

Check that a written plan solves the requested software problem, fits this
repository and can be implemented and validated without unnecessary machinery.
Perform the check directly. This skill does not require another agent or an
external review, and does not replace an independent review the user requested.
Review software behaviour, implementation scope and validation. Research
hypotheses, experiment variants, optimiser grids and result publication are
outside this skill.

## Establish the intended change

Read the plan and the relevant earlier conversation, including accepted
corrections, constraints and authorised actions. Read linked issues or PRs when
they supply missing requirements. If earlier discussion is unavailable, state
the gap rather than inventing requirements. Identify the intended behaviour,
scope and observable completion criteria before judging the proposed solution.

Read `AGENTS.md` / `CLAUDE.md` and use its reference table to select the relevant
`.claude/docs/` and `docs/` guidance. Inspect the named files, existing helpers,
tests and actual dependency pins. Read new untracked plan files explicitly;
they do not appear in `git diff`. Keep inspection proportional to the change.
When borrowing guidance from another repository, adapt it to this repository's
instructions; do not import its configuration, migration or publication workflow.

## Check repository fit and correctness

- Does the plan cover the user's actual request and all affected entry points,
  callers and consumers? Are sequencing, prerequisites and acceptance criteria
  concrete and consistent? Flag contradictory steps, circular prerequisites and
  success criteria that leave the original problem unresolved.
- Does it describe the current code correctly? Verify claimed APIs, selectors,
  configuration options, dependency contents and existing capabilities. Separate
  implemented behaviour from a proposed API, historical evidence and assumptions.
  A feature merged upstream may still be absent from the executor's pinned
  dependency; check both the adapter and its consumers before claiming support.
- Does it extend existing routing, state, configuration and recovery paths?
  For persistence or recovery changes, read
  [state persistence](../../../docs/state-persistence.md): use existing executor
  state, transaction records, state stores and callbacks. Flag a proposed second
  journal or checkpoint system unless the user explicitly authorised it.
- For vault integration, start from the
  [shared vault lifecycle guide](../../docs/vault-deposit-redeem.md) and its
  code/test map. Map requirements to existing managers, tickets, pending trades,
  settlement retry, valuation and funding gates before proposing changes.
  Keep external vault positions distinct from the strategy's Lagoon treasury.
  Require only missing doc/code cross-references needed to discover these paths;
  preserve useful existing links rather than writing another lifecycle guide.
- For execution or accounting changes, check actual amounts and units, ownership,
  fees, intermediate states, failure/retry behaviour and restart recovery. Check
  that pending proceeds are not spendable cash and repeated reconciliation does
  not apply an economic effect twice. Apply these checks where relevant rather
  than imposing transaction machinery on unrelated work.
- Check relevant repository conventions: CLI configuration at Typer entry points,
  strategy parameters, Python imports/types, naive UTC datetimes, token helpers,
  Web3 construction and robust receipt waits. Use the authoritative docs instead
  of copying their full rules into the plan. Include deployment, compatibility
  or migration work only when the actual change requires it.

## Check that validation proves the change

- Do the proposed checks exercise the changed behaviour through the real public
  entry point or shared implementation? Prefer a small happy-path and refusal or
  regression case over many tests that merely restate implementation details.
- Can the tests run with the stated environment, fixtures, dependency versions
  and fork block? Follow repository Poetry, private environment, worktree
  `PYTHONPATH`, parallelism and timeout rules. Missing prerequisites remain
  explicit gaps; do not promise a passing check that cannot run.
- For contract integration, validate real ABIs, calldata, receipts and custody
  restrictions. Use minimal Anvil impersonation/time changes and label them.
  Avoid spoofing the behaviour under test, injected liquidity that hides a real
  failure, or mocked events presented as integration evidence. An external
  indexer cannot discover newly mined local fork transactions.
- Check ABI provenance and the deployed proxy/implementation at the chosen
  block against the selected dependency. Use its shared ABI loaders and fork
  fixtures, including snapshot isolation, instead of inline interfaces or a
  new test harness. Check that the provider can serve the required historical
  state; do not assume every chain has arbitrary-depth archive access.
- For an external provider integration, include a minimal real-provider check
  when relevant to the changed behaviour; mocks alone do not establish provider
  compatibility. Use existing CI coverage or a focused manual command with
  credentials supplied through the normal environment. Record observed results
  with command, date and redacted provider details, distinguishing manual from
  CI evidence. Missing access is a validation gap, not permission to provision
  credentials, spend funds or publish a result.
- Cover meaningful interruption, failure and compatibility boundaries when the
  change affects persisted or on-chain operations. Distinguish failed attempts
  from successful but unresolved operations; never infer success from silence.
  Separate unit evidence, fork mechanics and observed live availability.

Review existing results and propose the focused checks needed. Do not implement
the plan or launch tests, builds, deployments or production changes as part of
this sanity check. Do not claim proposed validation has already passed.

## Remove unnecessary complexity

For each new abstraction, state model, dependency, configuration switch, fallback
or operational step, ask which concrete requirement it satisfies and whether an
existing mechanism or direct implementation already does the job. Remove or
defer speculative generality, duplicate sources of truth, redundant approvals,
unrelated cleanup and compatibility layers for unsupported hypothetical cases.
Include machinery hidden inside existing models in this check: new coordination
logic is still complexity even if it uses `other_data` rather than a new class.
Prefer a supported refusal for an exceptional case when it meets the agreed
scope; state any lost capability or deferred acceptance criterion explicitly.
Do not quietly narrow required behaviour to make the plan look simpler.

Look for avoidable multi-stage workflows, duplicated checks, brittle workarounds,
invented restrictions and tests requiring more machinery than the feature.
Propose the simpler route and explain why it still meets the acceptance
criteria. Keep safeguards that protect money, state, permissions and recovery;
simplicity is not a reason to discard a demonstrated invariant. Reuse upstream
test evidence for unchanged adapter behaviour and focus local tests on changed
consumers and public entry points. Do not turn the
sanity check itself into a framework, an exhaustive audit or a new review loop.

## Correct the plan and report

Update the identified plan directly to fix substantiated issues, preserving its
scope, useful evidence and established format. Honour an explicit review-only
request. Do not edit implementation files or unrelated plans. If a correction
depends on a missing requirement or unauthorised action, record that dependency
without treating it as approval or blocking independent corrections.

Check changed text for UK English, sentence-case headings, accurate local links
and clear distinctions between existing behaviour and proposed work.

Return the plan link and a concise table of issues, supporting evidence and
fixes made or still needed. If there are no issues, say so without manufacturing
findings. Identify material remaining assumptions or evidence gaps, and stop
once the plan is coherent and proportionate to the task.

## Sources and adaptation

- [freqtrade-strategies sanity-check-plan](https://github.com/tradingstrategy-ai/freqtrade-strategies/blob/42bdd12a68de35f3d9926f22047058826c24a836/.claude/skills/sanity-check-plan/SKILL.md):
  direct conversation/scope review, common-sense simplification, automatic plan
  corrections and an issues/fixes table. Research-specific checks were removed.
- [web3-ethereum-defi repository guidance](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/b26ac4b47aeb51bcb371dac1134e92f6e9c12d62/CLAUDE.md)
  and [shared Anvil testing guidance](https://github.com/tradingstrategy-ai/web3-ethereum-defi/blob/b26ac4b47aeb51bcb371dac1134e92f6e9c12d62/eth_defi/testing/README.md):
  real integration evidence, ABI provenance, provider limitations and reuse of
  existing test helpers. No `sanity-check-plan` skill was found in its remote
  master or inspected local worktrees on 2026-10-30; these documents supplied
  the web3 guidance, not a second copied skill.

These links record provenance. Use the current local repository instructions
and selected dependency's documentation when reviewing a plan; the source
repositories' research, scanner and PR-publication workflows are not adopted.
