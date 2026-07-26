# Testing

## Prerequisites

- Python 3.10
- Node 16 (prefer through NVM)
- Install with `poetry`
- Get [Trading Strategy API KEY](https://tradingstrategy.ai/trading-view/api)

Set up environment:

```shell

# We use production serverPlain to get datasets needed in tests
export TRADING_STRATEGY_API_KEY="" 

# We use BNB chain mainnet forking for some tests
export JSON_RPC_BINANCE="https://bsc-dataseed.binance.org/"

# ... and tons of other JSON RPCs for other chains
```

Set up Anvil:

```shell
# TODO foundryup
```

Make sure you install with the optional QSTrader dependency:

```shell
poetry install --a
```

## Running

To run the tests:

```shell
pytest 
```

## Running (parallel)

You need to use `loadscope` to parallerise the tests only on module level. 
Some fixtures cannot be parallerised between tests in the same module.

```shell
pytest --tb=native --dist loadscope -n 6
```

## Warm RPC fork tests

Fixed-block fork tests marked `warm_rpc_test_group` or
`warm_rpc_high_value_group` use eth-defi's Foundry RPC cache seeds. Read-only
and snapshot-safe mutating groups may share a worker-local `AnvilForkPool`;
mutating tests that need an isolated chain keep one Anvil process per test. Run
them with `loadgroup` so every matching `xdist_group` is serialised on one
worker:

```shell
source .local-test.env && PYTHONPATH="$(pwd):$(pwd)/deps/web3-ethereum-defi:$PYTHONPATH" \
  poetry run pytest -m warm_rpc_test_group -n auto --dist loadgroup --timeout=300
```

Test startup seeds the live Foundry cache non-destructively with eth-defi's
defaults and this repository's pinned fork entries in `tests/rpc_cache_seed/`.
An existing local cache is never replaced. Mutating warm-fork tests need either
a function-scoped `evm_snapshot_revert()` fixture when sharing Anvil or a
function-scoped isolated Anvil fixture. Refer to
`eth_defi.testing.anvil_fork_pool` for the pool contract and marker conventions.


## Interactive tests

Some tests provide interactivity. By default everything runs non-interactively.
But to test the user interface you might want to run the tests with user input enabled.

Tests that use this feature include
- `test_cli_approval`

```shell
USER_INTERACTION=true pytest
```

## Discord logging tests

You might want to test how Discord trade position mesages look like.

You can do it with:

```shell
# Webhook URL for the private trash channel
export DISCORD_TRASH_WEBHOOK_URL=...
export JSON_RPC_BINANCE="https://bsc-dataseed.binance.org/"
pytest -k test_pancake_4h_candles
```

This will execute 6 strategy cycles and log output.
