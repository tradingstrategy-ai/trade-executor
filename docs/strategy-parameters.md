# Strategy parameters

How a strategy module declares what it is and how it runs, for strategy
engine version `0.5`.

## One module-level variable

A strategy module declares exactly one module-level engine variable:

```python
trading_strategy_engine_version = "0.5"
```

Everything else lives in the `Parameters` class. The older module-level
variables are redundant for `0.5` and should not be declared:

| Old module variable | Why it is not needed |
|---|---|
| `trading_strategy_cycle` | Replaced by `Parameters.cycle_duration`, which the loader copies over it |
| `trade_routing` | Replaced by `Parameters.routing`, which the loader copies over it |
| `trading_strategy_type` | Defaults to `managed_positions`, the only supported type |
| `reserve_currency` | Live strategies name their reserve asset in `create_trading_universe()`; only the backtest runner reads this variable |

Module-level constants a strategy uses for its own universe, such as pair
lists, are fine. `strategies/xchain-master-vault.py` is the reference for
this layout. The production `strategies/lighter-ai.py` and the single-account
`strategies/test_only/minimal_lighter_strategy.py` predate it, read
`LIGHTER_ACCOUNT_INDEX` from the environment, and are left as they are.

## The `Parameters` class

`Parameters` is a plain class whose attributes the loader turns into a
`StrategyParameters` mapping, available to the strategy as
`input.parameters` in `decide_trades()` and `create_trading_universe()`,
and to commands as `mod.parameters`. Values are read by attribute or with
`parameters.get()`.

A parameter is something the strategy author decides about the strategy's
behaviour. It is not:

- **Deployment output.** Vault addresses, Lighter account indices, deployment
  blocks and similar facts are assigned at deployment. They live in the
  deployment artefact and in the executor state, which `init` fills. A
  strategy that needs them reads them from the state the universe input
  carries, never from `Parameters` or the environment.
- **Protocol constants.** Lighter's 1 USDC transfer minimum, token decimals or
  contract addresses belong in code constants, with sensible defaults in the
  functions that use them.
- **Test or tutorial knobs.** The amount a scripted test moves, a simulated
  fee or a timeout for a test fork belong in the test support code, next to
  the schedule that uses them.
- **Environment variables.** The executor's commands read the environment at
  their Typer entry points and pass settings down explicitly. A new strategy
  module must not read `os.environ`; a module that needs different values in
  different deployments declares them once and is deployed as different
  modules, or receives them from the state.

### Parameters the engine requires

| Parameter | Purpose |
|---|---|
| `cycle_duration` | How often `decide_trades()` runs, a `CycleDuration` |
| `routing` | Trade routing, normally `TradeRouting.default` |
| `initial_cash` | Starting capital of a backtest; required by the loader even for live-only strategies |
| `backtest_start`, `backtest_end` | Backtest range; `None` for live-only strategies |

### Parameters the engine reads when present

| Parameter | Purpose |
|---|---|
| `chain_id` | Chain the strategy trades on, or `ChainId.cross_chain` |
| `id`, `name` | Identity shown in diagnostics |
| `required_history_period` | Market data history loaded before the first cycle |
| `slippage_tolerance` or `slippage_tolerance_pct` | Maximum slippage per trade as a fraction; the first name wins when both are set |
| `max_price_impact` | Price impact limit for a trade |
| `stop_loss_pct`, `trailing_stop_loss_activation_level` | Position stop losses |
| `max_assets_in_portfolio`, `min_portfolio_weight`, `max_concentration`, `allocation` | Portfolio construction limits used by the alpha model helpers |
| `rebalance_threshold_usd`, `min_trade_size_usd`, `individual_rebalance_min_threshold_usd`, `sell_rebalance_min_threshold_usd` | Rebalance thresholds |
| `primary_chain_id`, `sweep_idle_bridge_capital`, `bridge_sweep_min_usd` | Cross-chain vault settings |
| `use_managed_yield`, `yield_flow_dust_threshold`, `directional_trade_yield_buffer_pct` | Yield management settings |

Any other attribute is the strategy's own and is read only by its
`decide_trades()`, `create_trading_universe()` and indicator functions.

### Exchange accounts and Lighter sub-accounts

A strategy that keeps capital on an external exchange declares the
sub-strategies it runs there, not the exchange's identifiers:

```python
from tradeexecutor.exchange_account.sub_account import SubAccount

class Parameters:
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    initial_cash = 100_000
    backtest_start = None
    backtest_end = None

    lighter_cash_management = True
    lighter_sub_accounts = [
        SubAccount(id="trend", name="Trend following", weight=0.5),
        SubAccount(id="arb", name="Funding-rate arbitrage", weight=0.5),
    ]
    lighter_safe_cash_buffer_usd = Decimal(20)
    lighter_free_collateral_buffer_usd = Decimal(0)
```

| Parameter | Purpose |
|---|---|
| `lighter_cash_management` | Enables Lighter custody routing in `start` |
| `lighter_sub_accounts` | The sub-accounts as `SubAccount(id, name, weight)` values. The id is the stable key the strategy moves funds with and the state maps to a Lighter account; never change it after deployment. The weight is the account's target share of the capital for the automatic cash policy |
| `lighter_safe_cash_buffer_usd` | USDC the automatic policy keeps in the Safe |
| `lighter_free_collateral_buffer_usd` | Free collateral the policy keeps on the master account |
| `lighter_min_transfer_usd` | Optional; defaults to Lighter's 1 USDC minimum |
| `lighter_rebalance_band_usd` | Optional tolerance before the policy moves capital between accounts; default 0 |
| `lighter_transfer_fee_usd` | Optional fee budget per transfer; default 0, as Lighter charges nothing between accounts of one owner |

Lighter assigns the account indices at deployment. `lagoon-deploy-vault`
creates one sub-account per declared `SubAccount`, `init` records the
indices in `state.sync.deployment.lighter_accounts`, and
`create_lighter_account_universe(input, Parameters.lighter_sub_accounts)`
builds the universe from the state. The strategy then moves funds by id
with `PositionManager.deposit_to_lighter()`, `withdraw_from_lighter()` and
`transfer_lighter_funds()`. See
[README-Lighter.md](../tradeexecutor/exchange_account/README-Lighter.md#strategy-setup-for-sub-accounts).

## Grid searches and optimisers

For a grid search or an optimiser, a parameter may be a list of candidate
values instead of a single value; the loader expands them into one parameter
set per combination. Keep the live module's `Parameters` single-valued and
run searches from notebooks, so a deployed module always has one
unambiguous configuration.
