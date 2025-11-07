"""REPL helper commands for CLI console."""
import datetime

from eth_typing import HexAddress

from web3 import Web3

from tqdm_loggable.auto import tqdm
from tabulate import tabulate

from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from eth_defi.erc_4626.core import get_vault_protocol_name, ERC4626Feature
from eth_defi.lagoon.vault import LagoonVault
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.token_cache import get_default_token_cache
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def list_vaults(
    console_context: dict,
):
    """List all vaults in the strategy universe, table formatted.

    Example:

    .. code-block:: python

        from tradeexecutor.cli.repl_utils import list_vaults
        list_vaults(locals())

    """
    web3 = console_context["web3"]
    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    vault_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.is_vault()]

    # Prepare vault instances to whitelist, and check
    # we can raad onchain data of them and there are not broken vaults/addresses

    token_cache = get_default_token_cache()

    vaults = []
    data = []
    broken_vaults = []
    for pair in tqdm(vault_pairs, desc="Processing vaults"):
        addr = pair.pool_address
        features = pair.get_vault_features()
        vault = create_vault_instance(
            web3,
            addr,
            features=features,
            token_cache=token_cache,
        )
        protocol_name = get_vault_protocol_name(vault.features)
        vaults.append(vault)
        data.append(
            {
                "Name": vault.name,
                "Address": addr,
                "Denomination": vault.denomination_token.symbol,
                "Protocol": protocol_name,
                "Features": ", ".join(f.value for f in vault.features),
            }
        )

        if ERC4626Feature.broken in vault.features:
            broken_vaults.append(vault)

    data = sorted(data, key=lambda x: x["Name"])

    # Display what we are about to whitelist
    table_fmt = tabulate(
        data,
        headers="keys",
        tablefmt="fancy_grid",
    )
    print("The following vaults are in the strategy universe:")
    print(table_fmt)



def deposit_4626(
    console_context: dict,
    address: HexAddress | str,
    amount_usd: float,
):
    """Make a command line trade to deposit into an ERC-4626 vault.

    Example:

    .. code-block:: python

        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            # Plutus hedge token
            "0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a",
            100.00,
        )

        # USDn2
        # 0x4a3f7dd63077cde8d7eff3c958eb69a3dd7d31a9
        deposit_4626(
            locals(),
            "0x4a3f7dd63077cde8d7eff3c958eb69a3dd7d31a9",
            100.00,
        )

        # Umami
        from tradeexecutor.cli.repl_utils import deposit_4626
        deposit_4626(
            locals(),
            "0x959f3807f0aa7921e18c78b00b2819ba91e52fef",
            100.00,
        )
    """

    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    pricing_model: PricingModel = console_context["pricing_model"]
    execution_model: EthereumExecution = console_context["execution_model"]
    routing_model: EthereumExecution = console_context["routing_model"]
    our_vault: LagoonVault = console_context["vault"]
    state: State = console_context["state"]
    web3: Web3 = console_context["web3"]
    store: JSONFileStore = console_context["store"]

    pair = strategy_universe.get_pair_by_smart_contract(address)
    vault = create_vault_instance_autodetect(
        web3,
        pair.pool_address,
    )
    web3 = console_context["web3"]

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
    )

    # Do guard checks
    trading_strategy_module = our_vault.trading_strategy_module
    trading_strategy_module_version = trading_strategy_module.functions.getTradingStrategyModuleVersion().call()

    print(f"Our vault is {our_vault.name} ({our_vault.fetch_version().value}) with TradingStrategyModule at {trading_strategy_module.address} ({trading_strategy_module_version})")

    print(f"About to deposit {amount_usd} USD to vault at {vault.name} ({pair.base.token_symbol} / {pair.quote.token_symbol})")
    print("Proceed [y/N]? ", end="")
    answer = input().strip().lower()
    if answer != "y":
        print("Aborting")
        return

    # Deposit to the vault
    trades = position_manager.open_spot(
        pair,
        value=amount_usd,
    )
    t = trades[0]

    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )

    assert t.is_success(), f"Trade failed: {t.get_revert_reason()}"

    store.sync(state)