# Preserve backtest scope when assessing live readiness

Read this before changing a backtest universe, converting a research strategy,
assessing vault readiness, or reviewing those changes. Live launch selection and
historical simulation answer different questions. Do not silently apply the
requirements of one to the other.

## The NB05 mistake

On 2 November 2026, the NB05 conversion incorrectly narrowed supported backtest
coverage while assessing production readiness:

- A launch-plan requirement for **current admission evidence** became a blanket
  historical selection rule: `deposit_permission != permissionless` meant
  exclusion. Twenty identities were removed before the existing historical
  admission rules could run. Unknown current metadata was treated as proof of
  historical prohibition.
- HYPE++ was excluded as an unverified D2 integration despite existing window,
  settlement and phase-aware backtest support. Its established assumed schedule
  was removed. HYPE++'s public eligibility predicate uses USDC itself; an
  additional funding-asset prerequisite was incorrectly inferred.
- A deposit end-date guard used either-direction settlement classification.
  NEMO's delayed exit therefore suppressed its synchronous deposit.
- Tests asserted the new exclusions rather than checking whether those
  exclusions followed the requested strategy and existing admission contract.
  Passing tests and independent code review did not detect that scope error.

The resulting 15.61% return described the changed universe, not a complete
comparison of the original strategy's supported routes. See the dated
[corrected backtest report](reports/nb05-eligibility-corrected-backtest-2026-11-02.md)
and its complete eligibility audit for identities, counts and evidence.
This lesson does not establish that every excluded vault
is admissible or ready for production.

## Rules for implementation

1. **Identify the mode and requested policy.** Keep historical simulation,
   execution capability, deployment validation and live owner admission
   distinct. A missing funded rehearsal or Guard check is not, by itself,
   evidence that an existing historical simulation cannot run.
   If the requested comparison is restricted to supported execution routes,
   retain labelled exclusions for genuinely missing execution, incompatible
   funding or payout assets. Simulation support alone does not satisfy that scope.
2. **Reuse existing admission rules.** Unknown is an evidence status, not a
   universal refusal or permission. Use the protocol-, timestamp- and
   operation-aware [vault closure contract](vault-closure.md). Do not bypass it
   with a strategy-wide current-metadata filter. Preserve explicit restrictions,
   fee completeness requirements, unsupported funding exclusions and applicable
   trading policy; do not fix over-exclusion by allowing everything.
3. **Verify a claimed missing capability.** Inspect prior fixes, the actual
   dependency pin, callers and relevant tests before declaring a family
   unsupported. Separate a missing implementation from incomplete deployment
   evidence, temporary closure, a data gap or an explicit simulation assumption.
4. **Preserve inherited assumptions unless changing them is authorised.** An
   existing assumed window is not proof of an actual historical schedule, but
   neither is it missing backtest support. Keep its label and provenance. A
   request not to add new synthetic windows does not itself authorise deleting
   an inherited one. Honour an explicit prohibition on all synthetic windows,
   but report the resulting scope change and loss of baseline comparability.
   Never use a current closed funding phase as proof of historical closure.
5. **Check directions and every allocation path.** A delayed redemption must
   not create a deposit queue or suppress an immediate deposit near the end of
   a backtest. Apply the appropriate admission policy to directional purchases,
   top-ups, parking and promotions. Preserve supported exits for existing holdings.

## Required review checks

For changes affecting selection, admission, timing or conversion, the primary
agent and independent reviewer must:

- Compare requested behaviour with both the baseline and final universe. Inspect
  new filters and removed configuration, including inherited schedules. For
  each new exclusion, identify its authorised policy, affected mode and
  identities, supporting evidence and existing mechanism being reused or bypassed.
  Use existing assessment/diagnostic output; no new registry is required.
- Check evidence dates. Today's unknown permission, blacklist, closure, capacity
  or owner restriction does not automatically describe the evaluation period.
  Retain an explicitly requested current-snapshot policy and label its limits.
- Independently derive expected admission from the user's rule and the shared
  contract before accepting a test expectation. A test that repeats a newly
  invented exclusion cannot establish that the exclusion is correct. Include
  appropriate evidence for a supported historical unknown/assumption case, an
  explicit refusal case, and deposit/redemption timing where affected.
- Examine original contributors removed or suppressed by the change. Explain
  material allocation and return differences before blaming unfinished protocols
  or fees. Separate ledger attribution from a matched counterfactual rerun;
  historical contribution is not guaranteed recoverable profit.
- Report unapproved narrowing as a correctness finding even if tests pass.
  If evidence is incomplete, record the precise gap rather than claiming a
  family is unsupported or the run fully covers supported routes.

These are review criteria, not a requirement to rerun experiments during a
read-only review. The implementing agent performs authorised fixes and focused
validation. Documentation of a mistake does not itself correct saved results or
the strategy code that produced them.
