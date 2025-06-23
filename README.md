[![Automated test suite and Docker image build](https://github.com/tradingstrategy-ai/trade-executor/actions/workflows/test-and-build-image.yml/badge.svg)](https://github.com/tradingstrategy-ai/trade-executor/actions/workflows/test-and-build-image.yml)

# Trade Executor: Algorithmic Trading Engine for DeFi

`trade-executor` is a Python framework for backtesting and live execution of algorithmic trading strategies on decentralised exchanges.

**Note**: This is early beta software. [Please pop in to the Discord for any questions](https://tradingstrategy.ai/community).

## Features

- Only trading framework that has been built grounds up for [decentralised finance](https://tradingstrategy.ai/glossary/decentralised-finance)
- [High quality documentation](https://tradingstrategy.ai/docs/)
- Support [decentralised markets like Uniswap, PancakeSwap](https://tradingstrategy.ai/docs/overview/supported-markets.html)
- [Backtesting engine](https://tradingstrategy.ai/docs/running/backtesting.html)
- [Live trading](https://tradingstrategy.ai/docs/running/live-trading.html)
- [Webhook web server](https://tradingstrategy.ai/docs/running/webhook.html) for JavaScript frontend and monitoring system integration
- Deploy as [Docker container](https://tradingstrategy.ai/docs/running/cli.html)

## Prerequisites

You need to know

- Basics of Python
- Basics of trading
- [We have collected learning material for developers new to algorithmic trading](https://tradingstrategy.ai/docs/learn/index.html)

## Getting started

First study the example code

- [Code examples](https://tradingstrategy.ai/docs/programming/code-examples/running.html)
- [Trading strategy examples](https://tradingstrategy.ai/docs/programming/code-examples/running.html)
- [See TradingView PineScript porting example](https://tradingstrategy.ai/blog/avalanche-summit-ii-workshop)

## More information

- [Read documentation on running and backtesting strategies](https://tradingstrategy.ai/docs/running/index.html)
- Visit [Trading Strategy website to learn about algorithmic trading on decentralised exchanges](https://tradingstrategy.ai)
- [Join the Discord for any questions](https://tradingstrategy.ai/community)

## Installation

**Note**: The project is under active development. We recommend any developers to use Github master branch
for installations.

As dependency line for Poetry `pyproject.yml`:

```toml
[tool.poetry.dependencies]
# Remove Python 3.11 pin down after upstream dependency issues are resolved
python = ">=3.10,<3.11"
# extras=["all"] does not seem to work here. Poetry bug?
trade-executor = {git = "https://github.com/tradingstrategy-ai/trade-executor.git", extras = ["web-server", "execution", "quantstats", "data"], rev = "master" }
```

Checking out from Github to make contributions:

```shell
git clone git@github.com:tradingstrategy-ai/trade-executor.git
cd trade-executor
git submodule update --init --recursive

# Extra dependencies
# - execution: infrastructure to run live strategies
# - web-server: support webhook server of live strategy executors
# - qstrader: still needed to run legacy unit tests
poetry install --all-extras
```

Or with pip:

```shell
pip install -e ".[web-server,execution,qstrader,quantstats]"
```

Or directly from Github URL:

```shell
pip install -e "git+https://github.com/tradingstrategy-ai/trade-executor.git@master#egg=trade-executor[web-server,execution,qstrader,quantstats]"
```

[**Limited file size by pre-commit hook**](scripts/pre-commit-sample/README.md)
```shell
# The pre-commit hook checks the size of files before allowing a commit to proceed
# If a file exceeds the specified limit, the commit will be aborted
# Default MAX FILE SIZE is 35MB
# Run script setup
cd trade-executor
bash scripts/set-pre-commit-checkfilesize.sh
```

## Architecture overview

Here is an example of a live trading deployment of a `trade-executor` package.

![Architecture overview](docs/deployment-overview.drawio.svg)

## Running tests

See [internal development documentation](https://tradingstrategy.ai/docs/programming/development.html).

## Community

- [Trading Strategy website](https://tradingstrategy.ai)
- [Community Discord server](https://tradingstrategy.ai/community#discord)
- [Blog](https://tradingstrategy.ai/blog)
- [Twitter](https://twitter.com/TradingProtocol)
- [Telegram channel](https://t.me/trading_protocol)
- [Newsletter](https://tradingstrategy.ai/newsletter)

## License

- AGPL
- [Contact for the commercial dual licensing](https://tradingstrategy.ai/about)
