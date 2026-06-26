# Deposit cases

This document lists the CCTP + async vault deposit cases this fix must cover
and how each case should be handled. It complements
`cctp-async-vault-accounting-plan.md`.

## Core rules

The strategy still works from global target equity. There is no
strategy-facing "deployable per-chain liquidity" concept.

CCTP bridge trades move capital between chains. Vault deposit trades consume
the vault pair's quote token on the vault chain. For a satellite vault, that
means the deposited token is satellite USDC (`pair.quote`), not the home-chain
reserve asset.

The CCTP planner should:

- size bridge-outs from the missing satellite-chain quote after existing idle
  satellite bridge capital is counted,
- cap bridge-backs to currently available bridge capital,
- ignore capital already committed to pending async vault deposits,
- ignore quote not yet received from pending async redeems,
- treat async vault sells as unavailable until their delayed settlement credits
  `pair.quote`, even if same-cycle satellite buys are also planned,
- keep sizing per satellite chain and never directly net satellite liquidity
  across chains.

Async vault backtesting should:

- debit `pair.quote` from the simulated wallet at deposit request time,
- treat the debit as reserve/quote moving into the vault deposit queue,
- record the exact queued quote amount on the trade for settlement,
- exclude queued deposit quote from spendable cash, bridgeable capital, and
  owner-wallet expected balances,
- leave `pair.base` shares at zero until settlement,
- keep the portfolio ledger's pending deposit value in equity,
- credit shares at settlement-cycle price,
- never debit quote again at settlement.

## Deposit cases

| Case | Setup | Handling |
|------|-------|----------|
| No idle satellite quote | Satellite vault buy is requested and no bridge position exists for that chain. | Inject a bridge-out for the full planned deposit amount. Execute bridge-out before the vault request. Debit satellite `pair.quote` at request. Pending wallet has no deposited quote and no shares. |
| Exact idle satellite quote | Existing available bridge capital equals the planned deposit amount. | Inject no bridge-out. Debit the exact idle satellite `pair.quote` at request. Primary reserve is unchanged. No residual satellite quote remains available. |
| Partial idle satellite quote | Existing available bridge capital is less than the planned deposit amount. | Inject one bridge-out for `planned_deposit - available_bridge_capital`. Because the shortfall is exact, no residual satellite quote remains after the request unless unrelated idle quote was also present. |
| Overfunded idle satellite quote | Existing available bridge capital is greater than the planned deposit amount. | Inject no bridge-out. Debit only the planned deposit amount at request. Residual satellite quote remains visible as available bridge capital and must pass wallet reconciliation. |
| Drained bridge position | A deposit exactly consumes all available bridge capital. | The bridge bookkeeping position may be closed or omitted from expected balances by the existing lifecycle, but wallet satellite quote and expected bridge asset quantity must both be zero. Tests should assert the wallet balance directly because zero-quantity assets may be absent from the expected map. |
| Multiple vault deposits on one satellite chain | Two or more async vault buys are planned on the same destination chain. | Aggregate required quote for the chain. Inject one bridge-out for the chain shortfall. Each vault request debits its own `pair.quote`; each pending deposit carries its own pending value. |
| Mixed spot buy and async vault deposit on one satellite chain | A satellite spot buy and satellite async vault buy are planned in the same cycle. | Aggregate the chain shortfall once. Effective backtest execution must run bridge-out before both buys. The spot buy settles normally; the vault buy debits quote at request and remains pending. |
| Multiple satellite chains | Async vault buys are planned on more than one satellite chain. | Size each chain independently. Idle or pending capital on one satellite cannot be directly netted against another satellite. Idle capital can only fund another satellite after it is bridged back through the primary chain and then bridged out to the target chain. Each bridge pair and wallet quote balance must reconcile per chain. |
| Pending deposit then bridge-back | A satellite async vault deposit is pending and a bridge-back is requested before it settles. | Bridge-back can use only available bridge capital. The test setup needs residual available quote or a realised satellite sell; quote committed to the pending deposit is not bridgeable and must not be counted by the planner. |
| Deposit settles after price drift | The vault price changes between request and settlement. | Pending value remains the committed quote amount. Settlement mints shares at the settlement-cycle price so settled share value equals committed quote value. No extra equity jump is introduced at claim. |
| Same-chain async vault deposit | The async vault lives on the primary chain and needs no CCTP bridge. | Apply the same request-time wallet debit rule using `pair.quote`. Existing same-chain delayed settlement and alpha-model pending guards must still pass. |
| Synchronous satellite vault deposit | A satellite vault is synchronous, not ERC-7540/Ostium-style async. | Keep existing synchronous execution: no `vault_settlement_pending` request stage and no async request-time wallet special case. CCTP bridge sizing is unchanged. |

## Redeem cases that affect deposits

Redeem handling matters because redeem proceeds can be mistaken for deposit
funding or bridge-back capital.

| Case | Setup | Handling |
|------|-------|----------|
| Pending satellite async redeem | A satellite vault redeem has requested settlement but not claimed quote. | Debit `pair.base` shares at request. Do not credit satellite `pair.quote` until settlement. Planner must not use pending proceeds for new deposits or bridge-backs. |
| Partial satellite async redeem | Only part of the vault share position is redeemed. | Escrow only redeemed shares. Remaining shares stay in the wallet and position. No new bridge capital appears until settlement credits quote. |
| Full satellite async redeem | The whole vault share position is redeemed. | Escrow the actual wallet share balance at request, including dust, and record that escrowed amount on the trade. Close the vault position only after settlement credits quote. The credited quote becomes available bridge capital after settlement. |
| Redeem plus same-chain satellite buy/deposit | A satellite async vault redeem and a same-chain satellite buy or deposit are planned in the same cycle. | Do not net the pending redeem against the buy/deposit. The buy/deposit needs existing idle bridge capital or a bridge-out. Redeem proceeds arrive only after settlement. |
| Redeem plus same-cycle bridge-back | A redeem is requested and a bridge-back is also planned in the same cycle. | Bridge-back is capped to pre-existing available satellite quote. Pending redeem proceeds cannot fund the same-cycle bridge-back. |
| Redeem settlement then later bridge-back | A pending redeem settles before a later cycle plans a bridge-back. | Settlement credits satellite `pair.quote`; a later bridge-back may move the newly available quote subject to normal bridge-capital accounting. |

## Reconciliation expectations

During a pending deposit:

- owner wallet should not hold the deposited `pair.quote`,
- the deposited `pair.quote` is in the vault deposit queue, not in reserves,
- owner wallet should not hold not-yet-minted `pair.base`,
- portfolio equity includes the pending deposit value,
- CCTP expected balances include only residual available bridge capital in both
  backtest wallet checks and account-correction paths.
- the bridge reconciliation contract is
  `wallet satellite pair.quote == available bridge capital == expected bridge asset quantity`,
  with zero available capital represented either as an omitted asset or an
  explicit zero.

During a pending redeem:

- owner wallet should not hold escrowed `pair.base`,
- owner wallet should not hold not-yet-claimed `pair.quote`,
- portfolio equity still values the share exposure until settlement,
- pending redeem value follows the current share price through the position
  valuation path, not through deposit-style pending value,
- CCTP expected balances must not include pending redeem proceeds.

After settlement:

- deposits credit only `pair.base` shares,
- redeems credit only `pair.quote`,
- settlement never repeats the request-time debit,
- deposit settlement uses the recorded request amount, not the residual wallet
  quote balance,
- redeem settlement uses the recorded escrowed amount, not the residual wallet
  share balance,
- wallet balances, bridge capital, position quantity, and total equity should
  all agree.
