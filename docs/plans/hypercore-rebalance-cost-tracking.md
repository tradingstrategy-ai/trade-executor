# HyperCore rebalance cost tracking

## Goal

Track and report the economic cost of normal HyperCore vault rebalances:

- HyperEVM transaction gas;
- HyperEVM ↔ HyperCore bridge fees;
- the one-off HyperCore account activation fee;
- observed value loss when fully closing a vault position.

Repaired, repair, test, unsuccessful, and transactionless trades are excluded
semantically. No trade IDs or date cut-offs are hardcoded.

## Persisted fields

`TradeExecution` uses its existing gas fields:

```python
cost_of_gas: Decimal | None
native_token_price: float | None
```

The following direct fields cover HyperCore-specific costs:

```python
bridge_input_amount: Decimal | None
bridge_output_amount: Decimal | None
bridge_fee_amount: Decimal | None
bridge_fee_asset: str | None
bridge_fee_usd: float | None
account_activation_fee_usd: float | None
hypercore_close_value_loss_usd: float | None
hypercore_close_other_loss_usd: float | None
hypercore_close_residual_value_usd: float | None
hypercore_cost_data_complete: bool = False
```

`None` means unknown or not measured. Zero means a measured zero. Existing
state files load without migration because every new field has a default.

## Measurement rules

### HyperEVM gas

After all trade transactions are confirmed, settlement sums:

```text
realised_gas_units_consumed × realised_gas_price
```

The result is stored in HYPE in `cost_of_gas`. A best-effort settlement-time
HYPE/USD mid price is stored in `native_token_price`. Missing gas receipt data
or a failed price query leaves the USD value unknown and never stops settlement.

Account activation is currently an exception: upstream `activate_account()`
does not return its approval and `depositFor` receipts. The first activated
deposit is therefore marked incomplete even when its other costs are measured.

### HyperEVM to HyperCore deposit

The bridge input is the USDC passed to the deposit escrow. HyperEVM-to-HyperCore
deposits have no protocol bridge fee, so a successfully verified deposit stores
an explicit zero fee. The bridge output is retained as telemetry: it is the
HyperCore spot USDC increase observed between the setup-time baseline and
escrow clearance, before spot capital is transferred into the vault.

The escrow wait has a tolerance, so a short observed spot increase is not
interpreted as a fee. Vault-equity movement is also not used to calculate a
deposit bridge fee because existing vault holdings can move during settlement.

A capped deposit is unused capital, not a fee.

### Account activation

Activation provisions 2 USDC. The executor measures HyperCore spot USDC before
and after activation:

```text
activation fee = provision - observed spot increase
```

Any USDC that reaches spot remains strategy capital. A zero observed increase
may mean the Info API has not caught up, so it leaves the activation fee unknown
instead of reporting the full provision as a fee.

### HyperCore to HyperEVM withdrawal

Before `sendAsset`, settlement records the Core spot balances and the USDC
principal passed to the bridge. After the Safe receives USDC, it reads the spot
balances again:

- a protocol-sized HYPE balance decrease is the HYPE-denominated bridge fee;
- otherwise, the spot USDC decrease beyond the principal is the USDC fee.

The 0.01 USDC amount reserved by
`compute_spot_to_evm_withdrawal_amount()` is safety headroom. It remains spot
capital and is not reported as a fee. Balance changes above the protocol-sized
sanity bounds remain unknown because they may be unrelated account activity.

### Full close loss

Only full closes get a close-loss measurement:

```text
close value loss =
    vault equity before
    - remaining vault equity after
    - USDC received on HyperEVM
    - phase-3 USDC headroom left in spot
```

The signed result can contain a vault performance commission and execution-
window NAV movement. It is described as observed value loss, not as an exact
performance fee.

Deposits have no equivalent entry-loss estimate. Beyond gas, activation, and
the fee-free deposit bridge, their vault-equity change is used for settlement
verification only because existing vault NAV can move during the confirmation
window.

For a USDC bridge fee, the fee is already included in the formula and is
subtracted into `hypercore_close_other_loss_usd`. A HYPE bridge fee leaves a
separate HYPE balance, so `hypercore_close_other_loss_usd` equals the USDC close
loss. Reporting always adds:

```text
bridge_fee_usd + hypercore_close_other_loss_usd
```

This includes the bridge fee exactly once. Partial withdrawals leave all close-
loss fields unknown because retained vault equity continues to move.

## Report

Run:

```shell
trade-executor show-hypercore-rebalance-costs \
    --state-file state/hyper-ai.json
```

Use `--details` to include the underlying trade table. The read-only command
prints:

1. one row per decision-cycle rebalance;
2. a whole-strategy summary;
3. ignored HyperCore history by semantic reason.

Exact total cost and BPS are shown only for complete rebalances:

```text
cost BPS = total economic cost / executed turnover × 10,000
```

Incomplete historic rows remain visible with `N/A`. Known historic gas is
reported separately as a lower bound using complete receipts and hourly HYPE
candle closes from the public Hyperliquid Info API.

## HyperAI historic result

The 2026-07-31 state snapshot contains:

- 80 eligible decision-cycle rebalances and 569 eligible trades;
- 418 trades with complete gas receipts;
- 2.537047 USD known gas cost;
- 185,347.554961 USD turnover covered by those gas receipts;
- 0.136881 BPS observed gas cost;
- 0.006069 USD average gas per receipt-complete trade.

Excluded history consists of 676 repair trades, 245 repaired originals, eight
test trades, three successful trades without transactions, and one
unsuccessful trade.

Historic bridge, activation, and close-loss fields were not persisted. The
historic gas result is therefore a lower bound, not the strategy's total cost.

## Costs deliberately not invented

| Component | Treatment |
|---|---|
| Vault performance commission | Included only in observed full-close loss |
| Vault NAV movement during close | Included only in observed full-close loss |
| Bridge safety headroom | Residual spot capital |
| Remaining vault dust | Residual vault capital |
| Capped deposit or withdrawal | Unexecuted or retained capital |
| Partial-withdrawal NAV delta | Unknown |
| Failed/repaired transaction gas | Excluded from the clean-rebalance report |
| LP fees | Not applicable to HyperCore vault transfers |

## Known limitation

To include activation gas, change upstream `activate_account()` to return the
two confirmed transaction receipts and attach them to the first buy trade.
Until then, activation deposits remain explicitly incomplete.
