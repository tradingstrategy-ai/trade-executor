# Testing

## Prerequisites

- Install with `poetry`

Set up environment:

```shell

# We use production server to get datasets needed in tests
export TRADING_STRATEGY_API_KEY="" 
```

## Running

To run the tests:

```shell
pytest 
```

## Interactive tests

Some tests provide interactivity. By default everything runs non-interactively.
But to test the user interface you might want to run the tests with user input enabled.

```shell
USER_INTERACTION=true pytest
```