"""Commoon Typer options shared across command line commands.

Import from there and use in
"""
from typing import Optional


from eth import Chain
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import HumanReadableTradingPairDescription
from typer import Option

id = Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance.\n"
                                                  "\n"
                                                  "If not given, take the base of --strategy-file.")

name = Option(None, envvar="NAME", help="Executor name used in the web interface and notifications")

log_level = Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing.")

strategy_file = Option(..., envvar="STRATEGY_FILE", help="Python trading strategy module to use for running the strategy")

private_key = Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet for paying transaction broadcasts and gas")

vault_address = Option(None, envvar="VAULT_ADDRESS", help="Deployed strategy vault address")
vault_adapter_address = Option(None, envvar="VAULT_ADAPTER_ADDRESS", help="Deployed GenericAdapter contract address for the vault")
vault_payment_forwarder = Option(None, envvar="VAULT_PAYMENT_FORWARDER_ADDRESS", help="USDC EIP-3009 payment forwarder contract for the Enzyme vault")

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

cache_path = Option("cache/", envvar="CACHE_PATH", help="Where to cache downloaded datasets on a local filesystem")

max_slippage = Option(0.0025, envvar="MAX_SLIPPAGE", help="Max slippage allowed per trade before failing. The default 0.0025 is 0.25% or 25 BPS.")

min_gas_balance = Option(0.1, envvar="MIN_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH. The default option 0.1 means that if your hot wallet account is less than 0.1 ETH the trade executor aborts via crash.")

test_evm_uniswap_v2_router: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_ROUTER",
                                                         help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")
test_evm_uniswap_v2_factory: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_FACTORY",
                                                          help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")
test_evm_uniswap_v2_init_code_hash: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_INIT_CODE_HASH",
                                                                 help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")


confirmation_block_count = Option(2, envvar="CONFIRMATION_BLOCK_COUNT", help="How many blocks we wait before we consider transaction receipt a final. Set to zero for automining testing backends (EthereumTester, Anvil).")

unit_testing = Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged.")

pair: Optional[HumanReadableTradingPairDescription] = Option(None, "--pair", envvar="PAIR", help="Must be specified for a multipair universe.")