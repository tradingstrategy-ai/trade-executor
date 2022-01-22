# Trade executor for Trading Strategy oracles

# Features

- Maintain the strategy state
- Listen to external events to perform a duty cycle
- Download historical datasets for the decision
- Run a strategy ticks

# Architecture

- Each strategy is deployed with an executor
- Each executor runs in its own process
- Strategy state can be serialized to a file or read from on-chain
- Executor open a WebHook port to accept ongoing signals

# Executable strategy definition

- Strategy must be a single Python module
- Strategy cannot have Python dependencies outside what `tradeexecutor` has
- Strategy module must export a class ``

# Execeutionn modes

# Running

Most of the options are passed as environment variables.

