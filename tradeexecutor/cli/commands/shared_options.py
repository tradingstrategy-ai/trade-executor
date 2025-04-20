"""Commoon Typer options shared across command line commands.

Import from there and use in
"""
import copy
import enum
from typing import Optional, Any

import typer
from typer import Option
from typer.models import OptionInfo




class PositionType(enum.Enum):
    """What output we want from show-positions command."""

    #: Show only open positions
    open = "open"

    #: Show only open positions
    open_and_frozen = "open_and_frozen"

    #: Show all positions
    all = "all"



def _gen_rpc_help(network_name: str):
    """Generate help text for RPC endpoint.

    """
    return f"This configures JSON-RPC endpoints for {network_name}. For the configuration format see https://web3-ethereum-defi.readthedocs.io/tutorials/multi-rpc-configuration.html"


def required_option(option: OptionInfo) -> OptionInfo:
    """Generate a copy of Typer option, but set it required."""
    new_option = copy.copy(option)
    new_option.default = None
    return new_option



id = Option(None, envvar="EXECUTOR_ID", help="Executor id used when programmatically referring to this instance.\n"
                                                  "\n"
                                                  "If not given, take the base of --strategy-file.")

name = Option(None, envvar="NAME", help="Executor name used in the web interface and notifications")


strategy_file = Option(..., envvar="STRATEGY_FILE", help="Python trading strategy module to use for running the strategy")
optional_strategy_file = Option(None, envvar="STRATEGY_FILE", help="Python trading strategy module to use for running the strategy")

private_key = Option(None, envvar="PRIVATE_KEY", help="Ethereum private key to be used as a hot wallet for paying transaction broadcasts and gas")

vault_address = Option(None, envvar="VAULT_ADDRESS", help="Deployed strategy vault address")
vault_adapter_address = Option(None, envvar="VAULT_ADAPTER_ADDRESS", help="Deployed GenericAdapter contract address for the vault")
vault_payment_forwarder = Option(None, envvar="VAULT_PAYMENT_FORWARDER_ADDRESS", help="USDC EIP-3009 payment forwarder contract for the Enzyme vault")
vault_deployment_block_number = Option(None, envvar="VAULT_DEPLOYMENT_BLOCK_NUMBER", help="When the vault was deployed: a block number before the deployment.")

asset_management_mode = Option(None, envvar="ASSET_MANAGEMENT_MODE", help="How does the asset management happen\n"
                                                                          ""
                                                                          "Hot wallet or vault based")


# Web3 connection options
json_rpc_binance = Option(None, envvar="JSON_RPC_BINANCE", help=_gen_rpc_help("RPC: BNB Smart Chain"))
json_rpc_polygon = Option(None, envvar="JSON_RPC_POLYGON", help=_gen_rpc_help("RPC: Polygon"))
json_rpc_ethereum = Option(None, envvar="JSON_RPC_ETHEREUM", help=_gen_rpc_help("RPC: Ethereum"))
json_rpc_avalanche = Option(None, envvar="JSON_RPC_AVALANCHE", help=_gen_rpc_help("RPC: Avalanche C-chain"))
json_rpc_arbitrum = Option(None, envvar="JSON_RPC_ARBITRUM", help=_gen_rpc_help("RPC: Arbitrum One"))
json_rpc_base = Option(None, envvar="JSON_RPC_BASE", help=_gen_rpc_help("RPC: Base"))
json_rpc_anvil = Option(None, envvar="JSON_RPC_ANVIL", help="Anvil JSON-RPC url. Anvil from Foundry is only used in local development and is not a readl blockchain.")

state_file = Option(None, envvar="STATE_FILE", help="JSON file where we serialise the execution state. If not given defaults to state/{executor-id}.json for live trade execution, state/{executor-id}-backtest.json for the backtest results.")

backtest_result = Option(None, envvar="BACKTEST_RESULT", help="State JSON file that contains the results of an earlier backtest run. Needed for the web server to display backtest information. If not given defaults to state/{executor-id}-backtest.json is assumed when the webhook server is started.")

notebook_report = Option(None, envvar="NOTEBOOK_REPORT", help="Jupyter Notebook file where the store the notebook results. If not given defaults to state/{executor-id}-backtest.ipynb.")
html_report = Option(None, envvar="HTML_REPORT", help="HTML file where the store the notebook results. If not given defaults to state/{executor-id}-backtest.html.")

trading_strategy_api_key = Option(None, envvar="TRADING_STRATEGY_API_KEY", help="Trading Strategy API key")

cache_path = Option(None, envvar="CACHE_PATH", help="Where to cache downloaded datasets on a local filesystem. If not given, default to cache/{executor-id}")

max_slippage = Option(None, envvar="MAX_SLIPPAGE", help="Legacy. Do not use. See slippagep.py details.")

min_gas_balance = Option(0.1, envvar="MIN_GAS_BALANCE", help="What is the minimum balance of gas token you need to have in your wallet. If the balance falls below this, abort by crashing and do not attempt to create transactions. Expressed in the native token e.g. ETH. The default option 0.1 means that if your hot wallet account is less than 0.1 ETH the trade executor aborts via crash.")

test_evm_uniswap_v2_router: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_ROUTER",
                                                         help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")
test_evm_uniswap_v2_factory: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_FACTORY",
                                                          help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")
test_evm_uniswap_v2_init_code_hash: Optional[str] = Option(None, envvar="TEST_EVM_UNISWAP_V2_INIT_CODE_HASH",
                                                                 help="Uniswap v2 instance paramater when doing live trading test against a local dev chain")


confirmation_block_count = Option(2, envvar="CONFIRMATION_BLOCK_COUNT", help="How many blocks we wait before we consider transaction receipt a final. Set to zero for automining testing backends (EthereumTester, Anvil).")
confirmation_timeout = Option(900, envvar="CONFIRMATION_TIMEOUT", help="How many seconds to wait for transaction batches to confirm")

unit_testing = Option(False, "--unit-testing", envvar="UNIT_TESTING", help="The trade executor is called under the unit testing mode. No caches are purged. Some special flags are set and checks are skipped.")

pair: Optional[str] = Option(None, "--pair", envvar="PAIR", help="A trading pair to be tested. Must be in format (chain_id, exchange_slug, base_token, quote_token, fee). Example: (base, uniswap-v3, KEYCAT, WETH, 0.0030).")

all_pairs: Optional[str] = Option(None, "--all-pairs", envvar="ALL_PAIRS", help="Whether to perform a test trade for each pair in the universe. If not given, then the pair option must be specified.")


comptroller_lib = Option(None, envvar="COMPTROLLER_LIB", help="Enzyme's ComptrollerLib address for custom deployments.")
simulate = Option(False, envvar="SIMULATE", help="Simulate transactions using Anvil mainnet fork. No new state file is written.")

# Log options
log_level = Option(None, envvar="LOG_LEVEL", help="The Python default logging level. The defaults are 'info' is live execution, 'warning' if backtesting. Set 'disabled' in testing.")
discord_webhook_url = Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications")
logstash_server = Option(None, envvar="LOGSTASH_SERVER", help="LogStash server hostname where to send logs")
file_log_level = Option("info", envvar="FILE_LOG_LEVEL", help="Log file log level. The default log file is logs/id.log.")
telegram_api_key = Option(None, envvar="TELEGRAM_API_KEY", help="Telegram bot API key for Telegram logging")
telegram_chat_id = Option(None, envvar="TELEGRAM_CHAT_ID", help="Telegram chat id where the bot will log. Group chats have negative id. Bot must receive command /start in the group chat before it can send messages there.")

max_workers = Option(None, envvar="MAX_WORKERS", help="Maximum number of worker processes (CPU cores) used for indicator calculations. Set to 1 for single thread mode to be used with Python debuggers. If not given use an autodetected safe value.")
position_type = Option("open_and_frozen", envvar="POSITION_TYPE", help="Which position types to display")


max_data_delay_minutes = typer.Option(24*60, envvar="MAX_DATA_DELAY_MINUTES", help="How fresh the OHCLV data for our strategy must be before failing")


def parse_comma_separated_list(ctx: typer.Context, value: str) -> list[str] | Any:
    """Support comma separated, whitespaced, list as a command line argument with Typer.

    Example:

    .. code-block:: python

        @app.command()
        def lagoon_deploy_vault(
            multisig_owners: Optional[str] = Option(None, callback=parse_comma_separated_list, envvar="MULTISIG_OWNERS", help="The list of acconts that are set to the cosigners of the Safe. The multisig threshold is number of cosigners - 1."),
        )

    """
    if ctx.resilient_parsing:
        return value
    if value is None:
        return []
    if value:
        return [item.strip() for item in value.split(',') if item.strip()]
    return []