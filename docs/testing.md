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
```

Set up Ganache:

```shell
npm install -g ganache
```

Make sure you install with the optional QSTrader dependency:

```
poetry install -E qstrader
```

## Running

To run the tests:

```shell
pytest 
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

