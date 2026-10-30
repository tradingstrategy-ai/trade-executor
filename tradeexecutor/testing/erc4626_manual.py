"""Read-only helpers for the Arbitrum Lagoon shell tutorial.

Protocol positions are opened and closed by public executor commands. This
module supplies universe and balance checks; existing Lagoon commands handle capital.
"""

import os
from decimal import Decimal
from pathlib import Path

import typer

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec

from tradeexecutor.cli.vault_test_trade import get_test_position
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import one_off_execution_context
from tradeexecutor.strategy.pandas_trader.create_universe_wrapper import call_create_trading_universe
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

STRATEGY_FILE = Path(__file__).resolve().parents[2] / "strategies/test_only/erc4626_manual_strategy.py"

# Match the main CLI: a failed RPC check must not print wallet locals.
app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)


def load_universe(strategy_file: Path = STRATEGY_FILE) -> TradingStrategyUniverse:
    """Load the tutorial's declared universe without remote datasets or environment reads."""
    module = read_strategy_module(strategy_file)
    return call_create_trading_universe(
        module.create_trading_universe,
        client=None,
        strategy_parameters=module.parameters,
        universe_options=module.get_universe_options(one_off_execution_context.mode),
        execution_context=one_off_execution_context,
    )


def position_phase(state: State, universe: TradingStrategyUniverse, address: str) -> str:
    """Choose the next leg from the ledger, never merely from a shell marker."""
    pair = next(p for p in universe.iterate_pairs() if p.pool_address == address.lower())
    position = get_test_position(
        state, pair,
        allow_pending_redemption=ERC4626Feature.atoma_like in (pair.get_vault_features() or set()),
    )
    if position is not None:
        return "close"
    closed = [p for p in state.portfolio.closed_positions.values() if p.pair == pair]
    if closed:
        if any(not t.is_test() for p in closed for t in p.trades.values()):
            raise RuntimeError("Tutorial state contains ordinary trades")
        if any(p.get_quantity() != 0 for p in closed):
            raise RuntimeError("Closed test position has residual shares")
        return "done"
    return "open"


@app.command()
def main(action: str, strategy_file: Path = STRATEGY_FILE, amount: str = "10", address: str | None = None) -> int:
    """Parse tutorial configuration once at the command entry point."""
    if action not in {"pairs", "identity", "preflight", "phase", "audit", "finished"}:
        raise typer.BadParameter("Action must be pairs, identity, preflight, phase, audit or finished")
    universe = load_universe(strategy_file)
    if action == "pairs":
        for pair in universe.iterate_pairs():
            print(f"{pair.get_vault_name()}\t{pair.pool_address}")
        return 0
    if action == "phase":
        if address is None:
            raise typer.BadParameter("phase requires --address")
        print(position_phase(State.read_json_file(Path(os.environ["STATE_FILE"])), universe, address))
        return 0
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    if web3.eth.chain_id != 42161:
        raise RuntimeError("This tutorial requires Arbitrum (42161)")
    wallet = HotWallet.from_private_key(os.environ["PRIVATE_KEY"])
    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    if action == "identity":
        print(wallet.address)
        return 0
    if action == "preflight":
        if usdc.fetch_balance_of(wallet.address) < Decimal(amount) or web3.eth.get_balance(wallet.address) == 0:
            raise RuntimeError("Investor needs the subscription USDC and signer gas")
        print(wallet.address)
        return 0
    vault = LagoonVault(web3, VaultSpec(42161, os.environ["VAULT_ADDRESS"]), trading_strategy_module_address=os.environ["VAULT_ADAPTER_ADDRESS"])
    if action in {"audit", "finished"}:
        state = State.read_json_file(Path(os.environ["STATE_FILE"]))
        if list(state.portfolio.get_open_and_frozen_positions()):
            raise RuntimeError("Positions remain open or unresolved; finish their execution before investor redemption")
        if any(position_phase(state, universe, p.pool_address) != "done" for p in universe.iterate_pairs()):
            raise RuntimeError("Not all protocol positions completed")
        if action == "finished" and vault.share_token.fetch_raw_balance_of(wallet.address):
            raise RuntimeError("Investor Lagoon shares remain; finish redemption")
        for pair in universe.iterate_pairs():
            if fetch_erc20_details(web3, pair.pool_address).fetch_raw_balance_of(vault.safe_address):
                raise RuntimeError(f"Residual protocol shares at {pair.pool_address}")
        print(f"Investor USDC balance: {usdc.fetch_balance_of(wallet.address)}")
    return 0


if __name__ == "__main__":
    app()
