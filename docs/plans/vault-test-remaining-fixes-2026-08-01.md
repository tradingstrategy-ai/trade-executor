# Alternative vault redemption simulations

## Purpose

When a production vault blocks redemption because of current liquidity or a
request window, the test matrix must retain that observation and may prove an
alternative lifecycle only on an Anvil fork. The alternative must use the real
adapter calls and receipts; it is not evidence that production is redeemable.

## Implemented paths

| Vault path | Natural result retained | Anvil-only preparation | Evidence required |
|---|---|---|---|
| Morpho Vault V2 transfer shortfall | `redemption_liquidity_unavailable` | Add the missing payout asset to the adapter or vault | Real redemption receipt and positive payout |
| Pharaoh USDC / 40acres | `redemption_capacity_limited` | Add direct-USDC until the address-scoped preflight accepts the request | Real unchanged receipt; payout is counterfactual because ERC-4626 assets change |
| Gains gTrade | `redemption_window_closed` | Advance to the epoch boundary and call permissionless `forceNewEpoch()` | Real request, settlement and claim receipts |

The common manager hook is
`prepare_redemption_simulation(owner, raw_shares, failure)`. It is unsupported
by default and only concrete protocol managers can opt in.

## Result reporting

The terminal row remains compatible with existing consumers:

```json
{
  "result": "success_simulated_with_intervention",
  "outcome_data": {
    "observed_result": "redemption_capacity_limited",
    "observed_detail": "40acres vault lacks immediate underlying liquidity for redemption",
    "interventions": [{"kind": "liquidity_injected"}]
  }
}
```

`observed_result` and `observed_detail` describe the natural production-state
refusal. `result=success_simulated_with_intervention` is emitted only after the
unchanged retry has completed through normal receipt analysis. Intervention
rows remain unsuitable for economic performance aggregates; a direct-USDC
injection can change a 40acres ERC-4626 payout.

## Safety rules

1. The natural request is attempted first.
2. Preparation runs only on Anvil and only after a typed, protocol-specific
   refusal.
3. The retry uses the same owner and share amount.
4. No live or manual execution injects assets, advances time or calls a
   protocol action.
5. The report records the original refusal and the synthetic token, time or
   permissionless transaction used by the fork. A funded ERC-4626 result is
   explicitly counterfactual, not a production price observation.
6. If preparation or the retry fails, retain the original typed result.

## Deliberately unresolved rows

These rows need separate source-level diagnosis before an alternative can be
claimed; they are not generic token-top-up candidates:

- Arche USD zero payout;
- YieldNest ynRWAx maturity plus zero immediate buffer;
- Ember minimum-share amount;
- Sentora accepted-asset lifecycle;
- Aerodrome satellite close amount; and
- DeTrade infrastructure timeout.

## Verification

- Run focused fork tests for Morpho, Pharaoh and Gains.
- Run `vault-test-trade --auto-simulated --settle-async-on-anvil` for the
  affected vault ids and inspect the machine report.
- Rerun the ordered 129-vault matrix after each batch of source-proven drivers.
