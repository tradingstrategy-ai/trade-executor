# Trade-executor cross-chain vault test report

## Evidence and scope

The earlier report was invalid for this branch: its persisted provenance named
trade-executor commit `42781eea`, and its tracebacks imported the parent
checkout, despite the report prose claiming `248e1e20`. It must not be used as
an adapter baseline.

The corrected 129-vault matrix ran on 2026-07-24 from this worktree with:

- trade-executor HEAD `2479c991`;
- eth-defi master `b5803bdc52606190969ca44af878b25cde8e3dec`;
- worktree-first imports for both `tradeexecutor` and the eth-defi submodule;
  and
- `--auto-simulated --settle-async-on-anvil`.

The run used uncommitted manager-owned settlement changes that were later
included in `9220345b`, but its JSON provenance records only the then-current
HEAD `2479c991`. Traceback paths and explicit import checks confirm the
worktree source was imported, but the exact dirty source state cannot be
reconstructed from the JSON. Treat the counts as observed behavioural evidence,
not as a commit-reproducible run. Operation attribution, share reconciliation
and capability-based async detection were added after this matrix snapshot and
have focused test coverage only.

## Corrected matrix output

| Status | Vaults | Interpretation |
|---|---:|---|
| Success (simulated) | 43 | Deposit and redemption lifecycle completed |
| Redemption pending | 6 | Four Ember and two Gains redemptions were not recognised as async from manager metadata |
| Deposit closed | 51 | Current on-chain admission state, principally Yearn `maxDeposit=0` |
| Whitelisting needed | 14 | The simulated executor Safe is not admitted |
| Adapter unsupported | 1 | Upshift multi-asset deposit is not implemented |
| Simulation unsupported async | 3 | Lagoon settlement ran but left the redemption ticket pending |
| Broadcast failed | 3 | Lagoon forced deposit settlement reverted for insufficient allowance |
| Transaction reverted | 5 | Two 40acres closes, cSigma redemption, YieldNest redemption and Accountable deposit |
| Execution failed | 2 | cSigma capacity assertion and Ember minimum-withdrawal error |
| Redemption unavailable | 1 | Plutus Hedge Token accepted the deposit but did not offer redemption |

The 51 closed and 14 whitelist-gated rows are useful admission results, not
adapter regressions. The 21 rows below are incomplete or unsupported lifecycle
coverage in this snapshot.

| Protocol | Vaults with gaps | Matrix result |
|---|---:|---|
| 40acres | 2 | Redemption transferred slightly more shares than the satellite module held |
| Accountable | 1 | Monad deposit reverted with undecoded custom error `0x5945ea56` |
| cSigma Finance | 2 | One `maxRedeem` capacity assertion; one `Withdrawal pending` redemption revert |
| Ember | 5 | Four asynchronous redemptions pending; Apollo ACRED below protocol minimum |
| Gains Network | 2 | Asynchronous redemptions pending |
| Lagoon Finance | 6 | Three tickets remained pending; three `settleDeposit()` allowance reverts |
| Plutus | 1 | Redemption currently unavailable |
| Upshift | 1 | Multi-asset deposit adapter unsupported |
| YieldNest | 1 | Redemption reverted with custom error `0xb8b8b59c` |

## Implemented trade-executor fixes

### Manager-owned forced settlement

`_force_vault_settlement_and_resolve()` now:

1. resolves the eth-defi manager and its capability;
2. reconstructs the persisted deposit or redemption ticket for the actual
   direction;
3. calls `manager.force_settle(ticket)` only when
   `supports_anvil_settlement is True`;
4. stores JSON-safe before/after statuses and settlement transaction hashes in
   `trade.other_data`;
5. retains the existing Ostium V1.5 permissionless Anvil compatibility path;
   and
6. raises `UnsupportedVaultSimulation` before broadcasting for every other
   manager.

This removes the signer-less Lagoon call. The corrected matrix contains no
`No Signer available`, `receipt_analysis_failed`, or
`state_inference_failed` rows. A focused Lagoon lifecycle completed both
deposit and redemption and persisted manager settlement evidence.

### Failure operation attribution

Automatic deposit attempts perform a deposit and, where possible, a
redemption. Failure capture now examines only new vault trades and labels the
attempt from the newest one. A failed sell is therefore reported as `redeem`
instead of inheriting the outer `deposit` label; later bridge trades cannot
overwrite it.

### Redemption share reconciliation

The old epsilon branch tested
`onchain_balance + planned_redemption < 0`. Both operands are positive, so it
was unreachable. The routing path now:

- retains the planned amount when the balance covers it;
- caps to the actual on-chain balance for a shortfall within the configured
  epsilon; and
- raises the accounting assertion before broadcasting for a material shortfall.

This addresses the two 40acres `transfer amount exceeds balance` rows. A
post-fix live rerun could not start because two fresh ephemeral Lagoon
deployments reverted during guard configuration, before either target vault
was executed; the deterministic reconciliation cases pass.

### Manager-declared async lifecycle detection

The batch runner now combines the pair kind with
`VaultDepositManagerCapability.deposit_flow` and `redemption_flow`. This
recognises mixed synchronous-deposit/asynchronous-redemption managers such as
Ember and Gains. With forced simulation enabled, their pending redemption now
reaches the existing direction-specific resolver and becomes
`simulation_unsupported_async` unless eth-defi advertises a settlement driver.
A successful synchronous deposit never enters forced settlement because the
resolver is gated by `TradeStatus.vault_settlement_pending`.

## Remaining trade-executor work

No protocol-specific settlement logic should be added to trade-executor.
Remaining executor work is verification and report provenance:

1. Repeat the full matrix after the ephemeral Lagoon deployment issue is
   resolved. Confirm the two 40acres rows close successfully, the six
   Ember/Gains rows become explicit unsupported-simulation results, and all
   redemption failures carry `operation=redeem`.
2. Record the imported eth-defi commit and dirty-worktree state in report
   provenance. The corrected JSON reports `eth_defi_commit=null`, and HEAD alone
   cannot identify uncommitted executor code.
3. Continue mapping typed eth-defi `UnsupportedVaultSimulation` and
   `VaultFlowUnavailable` exceptions to stable terminal results. Do not infer
   protocol policy from generic RPC assertion text.

The protocol-specific work required for completed simulations is listed in the
separate eth-defi report.

## Verification

Focused checks after implementation:

| Test | Result |
|---|---:|
| `tests/cli/test_vault_test_trade.py` | 44 passed |
| `tests/units_tests/test_home_chain_async_settlement.py` | 9 passed |
| `tests/units_tests/test_cross_chain_satellite_async_settlement.py` | 9 passed |
| `deps/web3-ethereum-defi/tests/lagoon/test_erc_7540_deposit_redeem.py` | 2 passed |
