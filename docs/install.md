# Installation

```shell
git clone git@github.com:tradingstrategy-ai/trade-executor.git
cd trade-executor
git submodule update --init --recursive
```

If you want to install with the legacy QSTrader support or if you are using a remote developement server:

```shell
poetry install -E qstrader -E web-server -E execution
```
otherwise, just use:

```shell
poetry install 
```
