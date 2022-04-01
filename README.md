# Trading Strategy API specifications

Decentralised trading data API in [OpenAPI 3 standard](https://swagger.io/specification/). 

**Beta warning**: Do not use at your own risk.

This repository contains OpenAPI 3 specifications for two APIs

* Open DeFi API for reading DEX data
* Trade Executor API for managing automated trading strategies

## Open DeFi  API

Open DeFi API provides server and browser accessible endpoints for decentralised exchange and blockchain live data. This data is useful for market data websites, real-time candel stick charts, chat bots, trading bots and similar.

[Open the Swagger API explorer](https://tradingstrategy.ai/api/explorer/).

The data covers

* Blockchains

* Exchanges

* Trading pairs

* OHLCV candles

* Available liquidity

## Trade Executor API

A trade executor runs live trading strategies as a server-side process.
It offers a webhook API to query its current state.

The data covers
 
* Current portfolio

* Open and closed positions

* Executed trades

* Profit and loss

* Deposit and withdraw events

* Internal metrics and diagnostics

**API explorer will be available later.**

## On-chain data oracle

**TODO**: The specification not yet available. [See Trading Strategy Python client for more information](https://tradingstrategy.ai/docs/programming/index.html).

Oracle data specification is intended for trading strategy backtesting, trading strategy programming, trading strategy oracles, trade instruction judge smart contracts and other on-chain logic. 

This specification covers

* API key accessible endpoints

* [Downloading backtesting datasets](https://tradingstrategy.ai/datasets)

# API directories

* [APIguru](https://github.com/APIs-guru/openapi-directory)
* [ProgrammableWeb](https://www.programmableweb.com/)
* [RapidAPI](https://rapidapi.com/hub)
* [IBM APIHarmony](https://apiharmony-open.mybluemix.net/public)

# Notes

- [Swagger online validator with human-readable errors](https://apitools.dev/swagger-parser/online/)

- [OpenAPI specification](https://swagger.io/specification/)

- [Pyramid example app](https://github.com/niteoweb/pyramid-realworld-example-app)

- [Counduit OpenAPI yaml from Pyramid example app](https://github.com/niteoweb/pyramid-realworld-example-app/blob/master/src/conduit/openapi.yaml)

- [Describing parameters](https://swagger.io/docs/specification/describing-parameters/)

- [Query page size example](https://github.com/Pylons/pyramid_openapi3/issues/155)

- [OpenAPI and arrays as query parameters (explode keyword)](https://swagger.io/docs/specification/serialization/)

- [OpenAPI and JSON query parameters](https://www.baeldung.com/openapi-json-query-parameters)

- [OpenAPI inter-file refs](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.0.md#referenceObject)

- [OpenAPI inter-file refs](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.0.md#referenceObject)
