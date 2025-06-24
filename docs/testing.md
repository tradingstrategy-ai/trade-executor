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
export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"

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
export BNB_CHAIN_JSON_RPC="https://bsc-dataseed.binance.org/"
pytest -k test_pancake_4h_candles
```

This will execute 6 strategy cycles and log output.

