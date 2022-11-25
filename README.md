[![.github/workflows/tests.yml](https://github.com/tradingstrategy-ai/trade-executor/actions/workflows/tests.yml/badge.svg)](https://github.com/tradingstrategy-ai/trade-executor/actions/workflows/tests.yml)

# Trade Executor

Trade Executor is a Python framework for executing algorithmic trading strategies on decentralised exchanges. 

**Note**: This is early alpha software. Please pop in to the Discord for any questions. 

## Features

- [High quality documentation](https://tradingstrategy.ai/docs/)
- Support [decentralised markets like Uniswap, PancakeSwap](https://tradingstrategy.ai/docs/overview/supported-markets.html) 
- [Live trading](https://tradingstrategy.ai/docs/running/live-trading.html) and [backtesting](https://tradingstrategy.ai/docs/running/backtesting.html)  
- [Webhook web serverPlain](https://tradingstrategy.ai/docs/running/webhook.html) for web and JavaScript integration
- Run the strategy execution as [Python application or Docker container](https://tradingstrategy.ai/docs/running/cli.html)

## More information

- [Read documentation on running and backtesting strategies](https://tradingstrategy.ai/docs/running/index.html)
- Visit [Trading Strategy website to learn about algorithmic trading on decentralised exchanges](https://tradingstrategy.ai)
- [Join the Discord for any questions](https://tradingstrategy.ai/community)

## Installation

```shell
git clone git@github.com:tradingstrategy-ai/trade-executor.git
cd trade-executor
git submodule update --init --recursive
poetry install -E web-server -E execution
```

## Architecture overview

![Archiecture overview](docs/deployment-overview.drawio.svg)

## Development

See [docs](./docs).

## Community

* [Trading Strategy website](https://tradingstrategy.ai)

* [Blog](https://tradingstrategy.ai/blog)

* [Twitter](https://twitter.com/TradingProtocol)

* [Discord](https://tradingstrategy.ai/community#discord) 

* [Telegram channel](https://t.me/trading_protocol)

## License 

- AGPL
