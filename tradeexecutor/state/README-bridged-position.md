# Cross-chain bridged position capital flow

This document explains how the trade executor handles capital accounting
when a strategy opens spot positions on a satellite chain using USDC that
was bridged there via Circle CCTP.

## Overview

A CCTP bridge position acts as a **virtual reserve** for the satellite
chain.  Home-chain reserves are never touched when trading on a satellite
chain — all capital flows through the bridge position instead.

```
  HOME CHAIN (Arbitrum)                          SATELLITE CHAIN (Base)
 +---------------------------+                  +---------------------------+
 |                           |                  |                           |
 |   Reserve: 5 USDC        |                  |                           |
 |                           |                  |                           |
 +---------------------------+                  +---------------------------+
              |                                              |
              | 1. bridge buy (depositForBurn)                |
              |  reserves -= 3                                |
              |                                              |
              |            CCTP attestation                   |
              |--------------------------------------------->|
              |                                              |
 +---------------------------+                  +---------------------------+
 |                           |                  |                           |
 |   Reserve: 2 USDC        |                  |   Bridge position: 3 USDC |
 |                           |                  |   (bridge_capital_alloc=0)|
 +---------------------------+                  +---------------------------+
                                                             |
                                                             | 2. spot buy WETH
                                                             |  bridge_capital_alloc += 2
                                                             |
                                                +---------------------------+
                                                |                           |
                                                |   Bridge position: 3 USDC |
                                                |   (bridge_capital_alloc=2)|
                                                |   WETH spot position: ~X  |
                                                +---------------------------+
                                                             |
                                                             | 3. spot sell WETH
                                                             |  bridge_capital_alloc -= 2.05
                                                             |
                                                +---------------------------+
                                                |                           |
                                                |   Bridge position: 3 USDC |
                                                |   (alloc=-0.05, PnL!)     |
                                                |                           |
                                                +---------------------------+
                                                             |
              |            CCTP attestation                   |
              |<---------------------------------------------|
              |                                              |
              | 4. bridge sell (reverse bridge 1 USDC)        |
              |  reserves += 1                                |
              |                                              |
 +---------------------------+                  +---------------------------+
 |                           |                  |                           |
 |   Reserve: 3 USDC        |                  |   Bridge position: 2 USDC |
 |                           |                  |                           |
 +---------------------------+                  +---------------------------+

  Total equity = 3 + 2 = 5 USDC (always conserved)
```

## Key data structures

### TradingPairKind.cctp_bridge

Defined in `identifier.py`.  A CCTP bridge pair has:

- **base** = USDC on the *destination* chain (where tokens are minted)
- **quote** = USDC on the *source* chain (where tokens are burned)
- **kind** = `TradingPairKind.cctp_bridge`

Helper methods:

| Method                      | Returns                         |
|-----------------------------|---------------------------------|
| `pair.get_source_chain_id()`      | `quote.chain_id` (burn chain)   |
| `pair.get_destination_chain_id()` | `base.chain_id` (mint chain)    |

### TradingPosition.bridge_capital_allocated

Field on `TradingPosition` (`position.py:320`).  Tracks how much of the
bridged USDC quantity is currently spoken for by satellite-chain trades.

```
available = position.get_quantity() - bridge_capital_allocated
```

Updated by `Portfolio.move_capital_from_bridge_to_spot_trade()` (increment)
and `Portfolio.return_capital_to_bridge()` (decrement).

### TradeExecution.bridge_currency_allocated

Field on `TradeExecution` (`trade.py:383`).  Records how much bridge
capital was allocated to this specific trade, analogous to
`reserve_currency_allocated` for normal trades.  Used by
`get_planned_reserve()` to report the allocated amount.

## Capital flow step by step

### 1. Bridge buy — home reserves → bridge position

When the strategy opens a CCTP bridge position (buy), `State.start_execution()`
deducts from the **home-chain reserves** exactly like a normal spot buy:

```python
# state.py:736-739
if trade.pair.is_cctp_bridge():
    if trade.is_buy():
        self.portfolio.move_capital_from_reserves_to_spot_trade(trade)
```

After execution succeeds the bridge position holds a quantity of USDC on
the destination chain (e.g. 3 USDC on Base).

### 2. Spot buy on satellite — bridge position → spot position

When the strategy opens a spot position (e.g. WETH-USDC) on the satellite
chain, `State.start_execution()` looks for an open bridge position
targeting that chain:

```python
# state.py:740-748
elif trade.is_spot():
    if trade.is_buy():
        bridge_position = self.portfolio.get_bridge_position_for_chain(
            trade.pair.chain_id,
        )
        if bridge_position is not None:
            self.portfolio.move_capital_from_bridge_to_spot_trade(trade)
        else:
            self.portfolio.move_capital_from_reserves_to_spot_trade(trade)
```

`get_bridge_position_for_chain()` (`portfolio.py:954`) iterates open
positions looking for one with `kind == cctp_bridge` whose
`get_destination_chain_id()` matches the trade's chain.

`move_capital_from_bridge_to_spot_trade()` (`portfolio.py:969`) does
**bookkeeping only** — no on-chain transfer:

1. Checks `bridge_position.get_available_bridge_capital() >= reserve`
2. Sets `trade.bridge_currency_allocated = reserve`
3. Increments `bridge_position.bridge_capital_allocated`

The home-chain reserves are **not** touched.

### 3. Spot sell on satellite — spot position → bridge position

When the satellite spot position is sold, proceeds go back to the bridge
position (not home reserves):

```python
# state.py:880-886
if trade.is_spot() and trade.is_sell():
    bridge_position = self.portfolio.get_bridge_position_for_chain(
        trade.pair.chain_id,
    )
    if bridge_position is not None:
        self.portfolio.return_capital_to_bridge(trade)
    else:
        self.portfolio.return_capital_to_reserves(trade)
```

`return_capital_to_bridge()` (`portfolio.py:1005`) decrements
`bridge_capital_allocated` by the executed reserve amount, making those
funds available for future satellite trades.

Note: `bridge_capital_allocated` can go negative when a profitable trade
returns more than was originally allocated.

### 4. Reverse bridge — bridge position → home reserves

Only when USDC is bridged **back** to the home chain (CCTP sell) do
proceeds return to source-chain reserves:

```python
# state.py:887-889
elif trade.pair.is_cctp_bridge() and trade.is_sell():
    self.portfolio.return_capital_to_reserves(trade)
```

This closes the bridge position and restores home-chain reserves.

## Equity calculation

`TradingPosition.get_equity()` (`position.py`) includes
`TradingPairKind.cctp_bridge` in its match case so that bridged USDC
counts towards total portfolio equity:

```python
match self.pair.kind:
    case (TradingPairKind.spot_market_hold
          | TradingPairKind.vault
          | TradingPairKind.freqtrade
          | TradingPairKind.exchange_account
          | TradingPairKind.cctp_bridge):
        return self.calculate_value_using_price(...)
```

Without this, bridged USDC would be invisible to portfolio valuation.

## Execution: transaction builder switching

When `GenericRouting.setup_trades()` (`generic_router.py`) encounters a
trade whose `pair.chain_id` differs from the home chain, it temporarily
swaps the `tx_builder` and web3 connection to the satellite chain.  For
Lagoon vaults this means creating a `LagoonTransactionBuilder` wrapping
the satellite `AutomatedSafe`; for hot-wallet strategies it creates a
plain `HotWalletTransactionBuilder` on the satellite web3.

The on-chain swap (e.g. Uniswap V3) executes against the USDC that was
physically minted on the satellite chain by CCTP — the routing layer does
not need to know where the USDC came from.

## Concrete example

Strategy deposits 5 USDC, bridges 3 to Base, swaps 2 for WETH, sells
WETH, bridges 1 USDC back.

```
Step  Action                     Home     Bridge   Bridge    Satellite
                                 reserve  qty      alloc     WETH
-----+---------------------------+--------+--------+---------+---------
  0  | Deposit                   |   5.00 |      — |       — |       —
  1  | Bridge 3 USDC -> Base     |   2.00 |   3.00 |    0.00 |       —
  2  | Buy 2 USDC of WETH        |   2.00 |   3.00 |    2.00 |      ~X
  3  | Sell WETH -> 2.05 USDC    |   2.00 |   3.00 |   -0.05 |       —
  4  | Bridge 1 USDC <- Base     |   3.00 |   2.00 |   -0.05 |       —
-----+---------------------------+--------+--------+---------+---------

  Total equity at any point = home reserve + bridge qty + satellite positions = 5.00
```

## File reference

| File | What it does |
|------|-------------|
| `state/identifier.py` | `TradingPairKind.cctp_bridge`, chain id helpers |
| `state/position.py` | `bridge_capital_allocated`, `get_available_bridge_capital()` |
| `state/trade.py` | `bridge_currency_allocated` on individual trades |
| `state/portfolio.py` | `get_bridge_position_for_chain()`, `move_capital_from_bridge_to_spot_trade()`, `return_capital_to_bridge()` |
| `state/state.py` | `start_execution()` routing logic, `mark_trade_success()` return logic |
| `strategy/generic/generic_router.py` | Transaction builder switching for satellite chains |
