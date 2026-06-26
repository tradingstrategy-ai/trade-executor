# CCTP async vault accounting plan

## Problem

PR #1543 fixed several symptoms in cross-chain CCTP + async vault backtests, but the remaining failure showed that the backtest wallet, portfolio ledger, and bridge planner still disagree during a satellite-chain async vault deposit.

The failing shape is:

1. Capital is bridged from the primary chain to a satellite chain with CCTP.
2. A satellite vault deposit is requested and the trade enters `vault_settlement_pending`.
3. Portfolio accounting treats the satellite USDC as committed to the vault request.
4. The simulated wallet still holds the satellite USDC until settlement.
5. The post-trade wallet reconciliation compares the wallet against open-position held assets and reports a mismatch.

The fix must not introduce a separate "deployable per-chain liquidity" model. Strategy target equity remains portfolio-wide: capital is moved between chains and positions by generated trades and their execution sort order.

## Facts to preserve

1. There should be no strategy-facing "deployable per-chain liquidity" concept. Target equity is global, and the trade sort mechanism handles the sequence of sells, bridge-backs, bridge-outs, and buys.
2. The CCTP planner should bridge only the missing amount needed to execute the generated satellite-chain buys after accounting for existing idle satellite bridge capital.
3. Async vault backtesting should make wallet reconciliation match the simulated vault lifecycle by debiting the simulated wallet at the vault request step, while settlement still follows the standard vault settlement delays and supported vault schedules added in earlier PRs.

## Design direction

Use request-time wallet movement for async vault deposits and redeems in backtests.

For a deposit request:

1. `state.start_execution()` allocates capital from reserves or the CCTP bridge position.
2. `simulate_async_vault_request()` marks the trade `vault_settlement_pending`.
3. The simulated wallet immediately debits the vault denomination token,
   `pair.quote`; this mirrors live behaviour where the reserve/quote balance
   leaves the owner wallet and enters the vault's deposit queue.
4. The debited reserve/quote is no longer spendable, bridgeable, or expected in
   owner-wallet reconciliation while the deposit is pending.
5. No vault shares are minted yet.
6. Portfolio equity remains continuous because the pending deposit value is carried by the `vault_settlement_pending` buy trade until shares settle.
7. Wallet reconciliation expects neither the debited `pair.quote` nor the not-yet-minted `pair.base` shares in the owner wallet during the pending window.
8. The delayed resolver later mints vault shares at the settlement-cycle price and marks the trade successful.

If the vault price moves before settlement, the pending deposit keeps carrying
the committed `pair.quote` value until claim. At settlement, the share quantity
is recalculated from the settlement-cycle price, so the settled share value
equals the committed quote amount and there is no artificial equity jump at
claim time.

For a redeem request:

1. The simulated wallet immediately debits the vault share token, `pair.base`, mirroring escrowed shares.
2. The position still tracks the shares until settlement, so portfolio equity remains continuous.
3. Existing expected-balance logic for pending redeems must keep subtracting escrowed shares where account checks need owner-wallet balances.
4. The delayed resolver later credits the denomination token, `pair.quote`, at the settlement-cycle price and marks the trade successful.

If the vault price moves before redeem settlement, the pending redemption
remains share exposure and should be valued using the same current-price
valuation path as a held vault position. At claim, the credited `pair.quote`
must match the settlement-cycle share value, so claim itself does not create an
extra accounting discontinuity beyond normal NAV movement.

This is closer to live ERC-7540 semantics than leaving request capital in the simulated wallet until claim.

The implementation must keep the portfolio ledger and simulated wallet ledger separate:

- `state.start_execution()` moves capital inside the portfolio accounting ledger only.
- `simulate_async_vault_request()` moves tokens in the simulated wallet ledger only.
- `simulate_async_vault_request()` also persists the exact queued amount on the
  trade: the debited `pair.quote` amount for deposits and the escrowed
  `pair.base` amount for redeems.
- Deposit settlement must not debit `pair.quote` again.
- Redeem settlement must not debit `pair.base` again.
- Settlement must only credit the token received at claim time and then record the trade success amounts needed by portfolio accounting.

## Coverage goals

Cover bridge and async-vault interactions by lifecycle stage, not just the original failing deposit case.

The detailed deposit/redeem case catalogue lives in
`.claude/docs/deposit-cases.md`. Keep that file aligned with this plan and use
it as the checklist when adding regression tests.

The key invariant across all cases is that physical wallet balances, portfolio accounting, and bridge bookkeeping describe the same state:

- pending deposits have spent `pair.quote`, have not minted `pair.base`, and keep equity continuous through pending value,
- pending redeems have escrowed `pair.base`, have not received `pair.quote`, and keep equity continuous through the existing position value,
- CCTP bridge positions expose only available satellite USDC as bridgeable capital, excluding capital already committed to async vault requests,
- the planner only bridges missing amounts and never invents a deployable-per-chain target model.

The target matrix should include:

| Case | Expected behaviour |
|------|--------------------|
| Bridge-out then async satellite deposit with no idle satellite capital | Bridge the full missing amount, then debit satellite `pair.quote` at vault request; pending wallet has no deposited USDC and no shares. |
| Async satellite deposit exactly matches idle satellite capital | Inject no bridge-out; debit exactly the existing satellite `pair.quote`; primary reserve stays unchanged and no residual satellite quote remains. |
| Async satellite deposit funded by partial idle satellite capital | Inject only the shortfall; because the shortfall is exact, no residual satellite quote remains after the request unless there was extra idle quote beyond the partial amount. |
| Async satellite deposit smaller than idle satellite capital | Inject no bridge-out; direct wallet assertions and reconciliation see only the residual available bridge capital in the wallet. |
| Multiple satellite buys on the same chain, including async vault and spot | Aggregate the chain shortfall once; effective execution runs bridge-out before both buys; async deposit spends quote at request while spot buy settles normally. |
| Multiple satellite chains with async vault buys | Size each chain independently; one chain's idle/pending capital must not fund another chain. |
| Satellite async deposit pending while bridge-back is requested | Bridge-back is capped to available bridge capital only; the setup must include residual bridge capital or a realised satellite sell because pending deposit capital cannot be bridged back. |
| Satellite async redeem request | Debit vault shares at request, do not credit satellite quote until settlement, and keep expected owner-wallet shares at zero while escrowed. |
| Partial satellite async redeem | Escrow only the redeemed shares; remaining shares stay in the wallet/position and bridge accounting stays unchanged until quote is credited. |
| Full satellite async redeem | Escrow all shares at request; close the vault position only after settlement credits quote, without losing bridge capital. |
| Satellite async redeem plus same-chain satellite buy/deposit | Do not net the async redeem sell against same-cycle satellite buys. New buys need idle bridge capital or a bridge-out; pending redeem proceeds arrive only after settlement. |
| Satellite async redeem plus same-cycle bridge-back | Do not bridge back redemption proceeds in the request cycle; only already-available satellite USDC may bridge back. |
| Satellite async redeem settlement followed by bridge-back | After settlement credits satellite `pair.quote`, later bridge-back can move only the newly available quote. |
| Same-cycle rotation from satellite async vault to primary or other satellite assets | New buys are capped to actually available cash/bridge capital; pending redeem proceeds are not counted. |
| Pending deposit settlement | Settlement credits shares at settlement-cycle price and does not debit quote again. |
| Pending redeem settlement | Settlement credits quote at settlement-cycle price and does not debit shares again. |
| Drifting-price pending deposit | Pending value stays at committed quote; settlement mints fewer or more shares at the settlement price without an equity jump. |
| Drifting-price pending redeem | Pending value follows the current share price; settlement credits quote at the same price without an extra claim-time jump. |
| Drained bridge position after exact deposit | When available bridge capital reaches zero, the bridge bookkeeping position may close or disappear from expected balances, but wallet and expected balances must both treat satellite quote as zero and direct wallet assertions must pin that edge case. |
| Synchronous satellite vault or spot trades | Existing CCTP bridge sizing and wallet behaviour stay unchanged. |
| Home-chain async vault deposit/redeem | Request-time wallet movement works without CCTP and existing same-chain async invariants still pass. |
| HyperCore-native vaults | Remain outside CCTP and ERC-7540/Ostium request-time wallet changes. |
| Underfunded bridge-out | Raise `NotEnoughMoney` early with the missing amount after idle satellite capital is deducted. |

## Implementation steps

1. Add cross-chain async vault regression coverage.

   Use `tests/ethereum/test_cctp_bridge_cash_aware.py` for planner and
   request-time wallet regressions unless the file becomes unwieldy. It already
   has the CCTP bridge-pair helpers and wallet/state setup needed for the new
   synthetic satellite vault pair. If the cases are split to a new
   `tests/ethereum/test_cctp_async_vault_accounting.py` module, move shared
   helpers into a `tradeexecutor/testing` module instead of importing from the
   tests tree.

   Start with a minimal failing sequence:

   - primary-chain USDC reserve,
   - CCTP bridge pair to a satellite USDC,
   - satellite async vault pair quoted in the satellite USDC,
   - bridge-out followed by an async vault deposit that remains pending.

   The test must assert the CCTP bridge-out has settled into the satellite
   simulated wallet before the vault request runs. Otherwise the test would
   exercise an unsettled CCTP failure instead of the async-vault accounting
   bug.

   Do not rely on `verify_balances()` alone: when available bridge capital is
   zero the expected map may not contain the satellite quote asset. Add direct
   wallet assertions for satellite `pair.quote` and `pair.base` after the
   request, and use the overfunded-idle case as the main residual bridge-capital
   reconciliation regression.

   Then extend the coverage around that helper:

   - no idle satellite capital: full bridge-out, pending deposit clean,
   - exact idle satellite capital: no bridge-out, no residual satellite quote,
   - partial idle satellite capital: bridge only the exact missing amount and
     leave no residual satellite quote,
   - overfunded idle satellite capital: leave residual available bridge capital
     in both wallet and expected balances,
   - mixed same-chain spot buy plus async vault buy: a single aggregate
     bridge-out, followed by normal spot settlement and pending vault request,
   - two satellite chains: independent bridge sizing and wallet reconciliation
     for each chain,
   - missing primary reserve: `NotEnoughMoney` reports the post-idle shortfall.

2. Add async redeem and bridge-back regression coverage.

   Extend the scenario helpers to open and settle an async satellite vault
   deposit, then request a redeem:

   - redeem request debits `pair.base` shares from the simulated wallet,
   - owner-wallet expected shares are zero while the redeem is pending,
   - satellite `pair.quote` is not credited until settlement,
   - same-cycle bridge-back is capped to pre-existing available satellite USDC,
     not the pending redeem proceeds,
   - same-cycle satellite buys/deposits are not funded by pending redeem
     proceeds; they need existing idle bridge capital or an injected bridge-out,
   - after redeem settlement, a later bridge-back can move the received
     satellite `pair.quote`,
   - partial redeem keeps the remaining share balance and bridge accounting
     correct,
   - full redeem closes the vault position without losing bridge capital.

3. Add settlement, pricing, and ordering assertions.

   For every cross-chain async case, pin the ordering and lifecycle explicitly:

   - effective backtest execution runs bridge-outs before bridge-dependent
     satellite buys/deposits, even if the raw sort key is later adjusted by the
     sequential CCTP execution path,
   - effective bridge-back execution happens before bridge-outs but can use only
     available bridge capital,
   - async deposits/redeems enter `vault_settlement_pending` instead of
     `success` on request,
   - settlement does not happen in the same cycle unless the existing backtest
     resolver rules allow it,
   - settlement uses settlement-cycle pricing,
   - settlement never repeats request-time wallet debits,
   - drifting-price deposit settlement keeps equity continuous by converting
     the committed quote into the settlement-price share quantity,
   - drifting-price redeem settlement keeps equity continuous by valuing the
     escrowed shares at the same price used for the credited quote,
   - pending-value accounting is direction-aware: pending deposits contribute a
     fixed committed-quote value through `get_vault_settlement_pending_value()`,
     while pending redeems do not add pending value and instead remain valued as
     share exposure through the position valuation path,
   - `calculate_total_equity()`, `get_vault_settlement_pending_value()`,
     bridge position quantity, bridge allocated capital, and wallet balances
     remain mutually consistent before request, during pending, and after
     settlement.

4. Change async request simulation.

   Update `BacktestExecution.simulate_async_vault_request()` so:

   - deposit requests debit `pair.quote` from the simulated wallet immediately,
   - the debit represents reserve/quote moving into the vault deposit queue and
     becoming unavailable until settlement,
   - redeem requests debit `pair.base` from the simulated wallet immediately,
   - the exact debited deposit quote and exact escrowed redeem share quantity
     are written to trade metadata for settlement to consume,
   - full redeem requests record the actual wallet share balance, including
     dust, so settlement closes the position cleanly,
   - partial redeem requests record the planned quantity after any request-time
     dust adjustment,
   - no shares/reserve proceeds are credited until delayed settlement.
   - the debit token is always derived from the vault pair, not from the home reserve currency.

5. Change async settlement simulation.

   Update `_settle_async_vault_trade()` so:

   - deposit settlement uses the recorded request reserve amount, not the
     current wallet `pair.quote` balance,
   - deposit settlement only credits `pair.base` shares,
   - deposit settlement removes the current settlement-time wallet quote
     dust-adjust/debit path; after request-time debit, wallet quote is residual
     capital, not the queued deposit,
   - redeem settlement uses the recorded escrowed request quantity, not the
     current wallet `pair.base` balance,
   - full redeem settlement uses the request-time actual wallet share balance
     recorded on the trade, not a stale planned quantity,
   - partial redeem settlement uses the request-time adjusted planned quantity
     recorded on the trade,
   - redeem settlement only credits `pair.quote` reserve,
   - settlement still uses the current settlement-cycle price,
   - trade success still records both executed quantity and executed reserve for portfolio accounting.
   - settlement never repeats the request-time wallet debit.

6. Make pending-request expected balances explicit.

   Ensure the expected-balance path used by the backtest wallet check and
   account-correction helpers models async pending requests symmetrically:

   - pending deposits: owner wallet should no longer hold the deposited
     `pair.quote`, and should not yet hold `pair.base` shares,
   - pending redeems: owner wallet should no longer hold escrowed `pair.base`
     shares, and should not yet hold the redeemed `pair.quote`.

   If the existing helpers already produce this outcome after request-time
   wallet movement, add assertions that pin the behaviour so it is not
   accidental.

   Add explicit expected-balance assertions for the bridge assets as well:

   - CCTP bridge positions should expect only available bridge capital in both
     backtest wallet checks and `check-accounts` / `correct-accounts`
     expected-balance paths,
   - the reconciliation contract is:
     `wallet satellite pair.quote == bridge_position.get_available_bridge_capital() == expected bridge asset quantity`,
     clamped/omitted as zero when the bridge position has no available capital,
   - `tradeexecutor.strategy.asset.get_asset_amounts()` must not expect the
     full gross CCTP bridge position quantity when part of it is allocated to
     satellite positions or pending async deposits,
   - implement this contract from available bridge capital directly rather than
     subtracting pending deposits independently in multiple places, to avoid
     double-subtracting queued quote,
   - committed async-deposit quote must be absent from both owner wallet and
     available bridge capital,
   - pending-redeem proceeds must be absent until settlement credits them.

7. Keep CCTP planner scope narrow.

   Preserve the planner responsibility from PR #1543:

   - bridge-outs are sized as the missing amount after existing idle satellite bridge capital,
   - bridge-backs are capped to capital that can actually be bridged back,
   - same-cycle async redeem proceeds are not treated as bridge-back capital or
     same-chain satellite-buy funding,
   - when computing satellite `chain_flows`, async vault sell trades must be
     distinguished from synchronous sells; pending async sells do not add
     spendable positive flow until settlement credits `pair.quote`,
   - bridge sizing is per destination chain and never netted across chains,
   - no target-sizing logic is moved into the planner,
   - no new deployable-per-chain abstraction is added.

8. Update existing tests.

   Update tests that currently assert delayed wallet movement for async deposits/redeems:

   - pending deposit: wallet reserve should be debited at request time and shares remain zero,
   - pending redeem: wallet shares should be debited at request time and reserve proceeds remain uncredited,
   - equity should remain continuous through the pending window.
   - same-chain async vault tests should still prove the delayed settlement
     lifecycle, current-price settlement, and alpha-model pending guards.
   - drifting-price lifecycle assertions belong in
     `tests/backtest/test_backtest_async_vault.py`,
   - account-correction / expected-balance assertions belong in
     `tests/ethereum/test_vault_settlement_lifecycle.py`,
   - planner and request-time CCTP wallet assertions belong in
     `tests/ethereum/test_cctp_bridge_cash_aware.py`.

   Before editing tests, grep for all delayed-wallet assumptions so the change
   is comprehensive:

   ```shell
   rg -n "wallet.*not.*debit|not debited|not credited|until claim|wallet_reserve|wallet_shares|simulated wallet" tests tradeexecutor .claude/docs
   ```

9. Update documentation.

   Update `.claude/docs/vault-deposit-redeem.md` so the backtesting section states:

   - the simulated wallet mirrors request-time token movement,
   - settlement delays still control when shares/proceeds are credited and trades become successful,
   - satellite-chain vault deposits spend `pair.quote`, not the home reserve currency.
   - CCTP bridge positions expose available satellite quote only; quote already
     committed to pending async deposits and quote not yet received from pending
     async redeems are not bridgeable.

   Add and maintain `.claude/docs/deposit-cases.md` as the focused checklist
   for CCTP + async vault deposit cases. It should describe each idle-capital,
   bridge, pending-deposit, redeem-interaction, settlement, and reconciliation
   case and how the implementation handles it.

## Verification

Run targeted tests only:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/ethereum/test_cctp_bridge_cash_aware.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/backtest/test_backtest_async_vault.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/backtest/test_cctp_backtest_sequential.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/ethereum/test_vault_settlement_lifecycle.py
source .local-test.env && PYTHONPATH="$(pwd):$PYTHONPATH" poetry run pytest tests/units_tests/test_cross_chain_satellite_async_settlement.py
```

Run `tests/ethereum/test_cctp_bridge_cash_aware.py` first because it is the
focused regression target. The `tests/units_tests` path is intentional and
already exists in this repository; run
`tests/units_tests/test_cross_chain_satellite_async_settlement.py` as an
existing cross-chain async control-flow regression, not because this fix creates
that file.

## Non-goals

- Do not add strategy-level per-chain deployable liquidity.
- Do not change alpha-model target equity semantics.
- Do not make the CCTP planner responsible for portfolio construction.
- Do not change live CCTP settlement semantics while fixing backtest wallet accounting.
- Do not change HyperCore-native vault bridge handling.
- Do not add rejected or failed async vault request simulation to backtests in
  this fix; the planned backtest coverage is successful delayed settlement.
- Do not run the whole test suite for this fix.
