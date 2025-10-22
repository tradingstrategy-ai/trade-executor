"""REPL tools for CLI console."""
import datetime

from eth_typing import HexAddress

from web3 import Web3

from eth_defi.erc_4626.classification import create_vault_instance, create_vault_instance_autodetect
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def list_vaults():
    pass


def deposit_4626(
    console_context: dict,
    address: HexAddress | str,
    amount_usd: float,
):
    """Make a command line trade to deposit into an ERC-4626 vault."""

    strategy_universe: TradingStrategyUniverse = console_context["strategy_universe"]
    pricing_model: PricingModel = console_context["pricing_model"]
    execution_model: EthereumExecution = console_context["execution_model"]
    routing_model: EthereumExecution = console_context["routing_model"]
    state: State = console_context["state"]
    web3: Web3 = console_context["web3"]

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

    print(f"About to deposit {amount_usd} USD to vault at {vault.name} ({pair.symbol})")
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
