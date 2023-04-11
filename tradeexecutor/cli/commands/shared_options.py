"""Commoon Typer options shared across command line commands.

Import from there and use in
"""
from typing import Optional


from eth import Chain
from tradingstrategy.chain import ChainId
from typer import Option

id = Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance.\n"
                                                  "\n"
                                                  "If not given, take the base of --strategy-file.")

log_level = Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing.")

strategy_file = Option(..., envvar="STRATEGY_FILE", help="Python trading strategy module to use for running the strategy")

private_key = Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet for paying transaction broadcasts and gas")

vault_address = Option(None, envvar="VAULT_ADDRESS", help="Deployed strategy vault address")
vault_adapter_address = Option(None, envvar="VAULT_ADAPTER_ADDRESS", help="Deployed GenericAdapter contract address for the vault")

asset_management_mode = Option(None, envvar="ASSET_MANAGEMENT_MODE", help="How does the asset management happen\n"
                                                                          ""
                                                                          "Hot wallet or vault based")


# Web3 connection options
json_rpc_binance = Option(None, envvar="JSON_RPC_BINANCE", help="BNB Chain JSON-RPC node URL we connect to")
json_rpc_polygon = Option(None, envvar="JSON_RPC_POLYGON", help="Polygon JSON-RPC node URL we connect to")
json_rpc_ethereum = Option(None, envvar="JSON_RPC_ETHEREUM", help="Ethereum JSON-RPC node URL we connect to")
json_rpc_avalanche = Option(None, envvar="JSON_RPC_AVALANCHE", help="Avalanche C-chain JSON-RPC node URL we connect to")
json_rpc_arbitrum = Option(None, envvar="JSON_RPC_ARBITRUM", help="Arbitrum C-chain JSON-RPC node URL we connect to")
json_rpc_anvil = Option(None, envvar="JSON_RPC_ANVIL", help="Anvil JSON-RPC url. Anvil from Foundry is only used in local development and is not a readl blockchain.")

state_file = Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json")

trading_strategy_api_key = Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key")