"""Helper commands for ``scripts/lagoon/manual-multi-vault-cli.sh``.

The shell script performs every lifecycle action with ``trade-executor``
commands. This helper only answers questions from executor state and the
chain, and manages the simulated environment:

- ``simulate-start`` / ``simulate-stop``: the forks, fake Lighter sequencer
  and forged CCTP of :py:mod:`tradeexecutor.testing.multi_vault_simulation`;
- ``cli -- <trade-executor arguments>``: run one registered executor command
  in this process, with the simulation installed when
  ``MULTI_VAULT_SIM_DIR`` is set;
- ``identity``, ``preflight``, ``targets``, ``phase``, ``audit`` and
  ``finished``: read-only checks that decide the next shell step;
- ``atoma-unlock``: simulation only, advance the Arbitrum fork's clock past
  Atoma's share lock (a labelled time spoof; the epoch settlement itself is
  the existing Anvil test driver of ``perform-test-trade``).

It writes no progress files: the next step is derived from the state file.
"""

import calendar
import logging
import os
import sys
from decimal import Decimal
from pathlib import Path

import typer
from eth_defi.erc_4626.vault_protocol.atoma.vault import AtomaVault
from eth_defi.erc_4626.vault_protocol.lagoon.lighter import LIGHTER_MULTICHAIN_BOOTSTRAP_SUBSCRIPTION
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import is_anvil, mine
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from tabulate import tabulate
from tradingstrategy.chain import ChainId
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app as executor_app
from tradeexecutor.ethereum.lighter.lighter_pool import create_lighter_pool_session
from tradeexecutor.ethereum.lighter.lighter_pool_valuation import LighterPoolPricing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.erc4626_manual import load_universe
from tradeexecutor.testing.lighter_manual import LAGOON_DEPOSIT_REQUEST_ID, broadcast_and_wait, get_vault
from tradeexecutor.testing.multi_vault_simulation import (
    MultiVaultSimulation,
    installed_multi_vault_simulation,
    start_multi_vault_simulation,
    stop_multi_vault_simulation,
)

logger = logging.getLogger(__name__)

#: The tutorial's strategy module.
STRATEGY_FILE = Path("strategies/test_only/multi_vault_arbitrum.py")

#: Minimum native gas per chain before a live run, in ETH.
MINIMUM_GAS = {42161: Decimal("0.005"), 1: Decimal("0.02")}

# Match the main CLI: a failed RPC check must not print wallet locals.
app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)


def get_target_pair(universe: TradingStrategyUniverse, target: str) -> TradingPairIdentifier:
    """Return the pair tested for ``lighter``, ``atoma`` or ``upshift``.

    :param universe:
        Tutorial universe.
    :param target:
        Target name.
    :return:
        The single matching vault pair.
    """
    if target == "lighter":
        matches = [p for p in universe.iterate_pairs() if p.is_lighter_pool()]
    else:
        matches = [p for p in universe.iterate_pairs() if p.is_vault() and not p.is_lighter_pool() and p.get_vault_protocol() == target]
    assert len(matches) == 1, f"Expected one {target} pair, found {len(matches)}"
    return matches[0]


def get_target_phase(state: State, pair: TradingPairIdentifier) -> str:
    """Choose a target's next leg from the ledger.

    :return:
        ``open``, ``pending``, ``close`` or ``done``.
    :raise RuntimeError:
        For ordinary positions, failed trades or a closed position with residual shares.
    """
    positions = [p for p in state.portfolio.get_all_positions() if p.pair == pair]
    for position in positions:
        if any(not t.is_test() for t in position.trades.values()):
            raise RuntimeError(f"{pair.get_ticker()} has ordinary strategy trades; this tutorial uses only test positions")
        if any(t.is_failed() for t in position.trades.values()):
            raise RuntimeError(f"{pair.get_ticker()} has a failed trade; inspect the state before resuming")
    if not positions:
        return "open"
    if any(t.get_status() in {TradeStatus.vault_settlement_pending, TradeStatus.started, TradeStatus.broadcasted} for p in positions for t in p.trades.values()):
        return "pending"
    if any(p.is_open() for p in positions):
        return "close"
    if any(p.get_quantity() != 0 for p in positions):
        raise RuntimeError(f"Closed {pair.get_ticker()} position has residual quantity")
    # A satellite exit is complete only once its proceeds are bridged back.
    # The bridge position records which vault's redemption it is returning.
    bridge = state.portfolio.get_bridge_position_for_chain(pair.get_funding_chain_id())
    redemption = bridge.other_data.get("vault_test_satellite_redemption", {}) if bridge is not None else {}
    if redemption.get("vault_id") == f"{pair.chain_id}-{pair.pool_address}":
        bridge_back = bridge.trades.get(redemption.get("bridge_back_trade_id"))
        if bridge_back is None:
            return "close"
        if not bridge_back.is_success():
            return "pending"
    return "done"


def get_simulation() -> MultiVaultSimulation | None:
    """Return the running simulation, or ``None`` in live mode."""
    directory = os.environ.get("MULTI_VAULT_SIM_DIR")
    return MultiVaultSimulation.load(Path(directory)) if directory else None


@app.command("simulate-start")
def simulate_start(directory: Path = typer.Option(..., envvar="MULTI_VAULT_SIM_DIR"), deployer_usdc: str = typer.Option("131")) -> None:
    """Start the forks and fake venue, then print the fork RPC exports for ``env.sh``."""
    simulation = start_multi_vault_simulation(directory, deployer_usdc=Decimal(deployer_usdc))
    print(f"export JSON_RPC_ARBITRUM={simulation.arbitrum_rpc}")
    print(f"export JSON_RPC_ETHEREUM={simulation.ethereum_rpc}")


@app.command("simulate-stop")
def simulate_stop(directory: Path = typer.Option(..., envvar="MULTI_VAULT_SIM_DIR")) -> None:
    """Stop the forks; the directory is kept for inspection."""
    stop_multi_vault_simulation(directory)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cli(ctx: typer.Context) -> None:
    """Run one ``trade-executor`` command in this process, simulated when configured."""
    command = get_command(executor_app)
    simulation = get_simulation()
    code = 0
    try:
        if simulation is None:
            command.main(args=list(ctx.args), standalone_mode=False)
        else:
            with installed_multi_vault_simulation(simulation):
                command.main(args=list(ctx.args), standalone_mode=False)
    except SystemExit as error:
        code = error.code or 0
    if code:
        sys.exit(code)


@app.command()
def identity() -> None:
    """Print the deployer address."""
    print(HotWallet.from_private_key(os.environ["PRIVATE_KEY"]).address)


@app.command()
def targets(strategy_file: Path = typer.Option(STRATEGY_FILE)) -> None:
    """Print ``target<TAB>--pair value`` for each tested vault."""
    universe = load_universe(strategy_file)
    for target in ("lighter", "atoma", "upshift"):
        pair = get_target_pair(universe, target)
        print(f"{target}\t({ChainId(pair.chain_id).get_slug()}, {pair.pool_address})")


@app.command()
def preflight(
    subscription_usdc: str = typer.Option("0"),
    bootstrap: bool = typer.Option(False, help="Also require the 20 USDC that deployment subscribes."),
    strategy_file: Path = typer.Option(STRATEGY_FILE),
) -> None:
    """Refuse to start without funds, gas or an open Lighter pool."""
    wallet = HotWallet.from_private_key(os.environ["PRIVATE_KEY"])
    simulation = get_simulation()
    rows = []
    for chain_id, variable in ((42161, "JSON_RPC_ARBITRUM"), (1, "JSON_RPC_ETHEREUM")):
        web3 = create_multi_provider_web3(os.environ[variable])
        assert web3.eth.chain_id == chain_id, f"{variable} is chain {web3.eth.chain_id}, expected {chain_id}"
        gas = Decimal(web3.eth.get_balance(wallet.address)) / Decimal(10**18)
        rows.append((variable, "ETH", gas, MINIMUM_GAS[chain_id]))
        if gas < MINIMUM_GAS[chain_id]:
            raise RuntimeError(f"Deployer {wallet.address} has {gas} ETH on chain {chain_id}; needs {MINIMUM_GAS[chain_id]}")
    arbitrum = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    usdc = fetch_erc20_details(arbitrum, USDC_NATIVE_TOKEN[42161])
    required = (LIGHTER_MULTICHAIN_BOOTSTRAP_SUBSCRIPTION if bootstrap else Decimal(0)) + Decimal(subscription_usdc)
    balance = usdc.fetch_balance_of(wallet.address)
    rows.append(("Arbitrum USDC", "USDC", balance, required))
    if balance < required:
        raise RuntimeError(f"Deployer {wallet.address} has {balance} Arbitrum USDC; needs {required} for {'the 20 USDC bootstrap and ' if bootstrap else ''}the subscription")
    pair = get_target_pair(load_universe(strategy_file), "lighter")

    def check_pool():
        return LighterPoolPricing(create_lighter_pool_session()).check_deposit(None, pair)

    if simulation is None:
        result = check_pool()
    else:
        with installed_multi_vault_simulation(simulation):
            result = check_pool()
    rows.append((pair.pool_address, "pool open", result.can_deposit, True))
    print(tabulate(rows, headers=["Check", "Unit", "Observed", "Required"]))
    if not result.can_deposit:
        raise RuntimeError(f"Lighter pool {pair.pool_address} does not accept deposits: {result.message}. Choose the fallback pool deliberately in the strategy module.")


@app.command("request-deposit")
def request_deposit(amount: str = typer.Option(..., envvar="SUBSCRIPTION_USDC")) -> None:
    """Investor's part of a Lagoon subscription: request a deposit unless one waits.

    Prints ``requested``, ``pending`` or ``claimable``; the status comes from
    the vault, so a rerun never requests twice while one is outstanding.
    """
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    wallet = HotWallet.from_private_key(os.environ["PRIVATE_KEY"])
    vault = get_vault(web3)
    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    if vault.vault_contract.functions.pendingDepositRequest(LAGOON_DEPOSIT_REQUEST_ID, wallet.address).call() > 0:
        print("pending")
        return
    if vault.vault_contract.functions.maxDeposit(wallet.address).call() > 0:
        print("claimable")
        return
    balance = usdc.fetch_balance_of(wallet.address)
    if balance < Decimal(amount):
        raise RuntimeError(f"Investor {wallet.address} has {balance} Arbitrum USDC; needs {amount} for the subscription")
    amount_raw = usdc.convert_to_raw(Decimal(amount))
    broadcast_and_wait(web3, wallet, usdc.approve(vault.address, Decimal(amount)), f"approve {amount} USDC to the vault")
    broadcast_and_wait(web3, wallet, vault.request_deposit(wallet.address, amount_raw), f"request a {amount} USDC deposit")
    print("requested")


@app.command("claim-shares")
def claim_shares() -> None:
    """Investor claims a settled deposit as shares; a no-op when nothing is claimable."""
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    wallet = HotWallet.from_private_key(os.environ["PRIVATE_KEY"])
    vault = get_vault(web3)
    if vault.vault_contract.functions.maxDeposit(wallet.address).call() > 0:
        broadcast_and_wait(web3, wallet, vault.finalise_deposit(wallet.address), "claim the settled deposit as shares")
    elif vault.vault_contract.functions.pendingDepositRequest(LAGOON_DEPOSIT_REQUEST_ID, wallet.address).call() > 0:
        raise RuntimeError("The deposit request is not settled yet; run trade-executor lagoon-settle first")
    print(f"Investor holds {vault.share_token.fetch_balance_of(wallet.address)} {vault.share_token.symbol}")


@app.command()
def phase(target: str, strategy_file: Path = typer.Option(STRATEGY_FILE)) -> None:
    """Print the next leg of a target from the state file."""
    pair = get_target_pair(load_universe(strategy_file), target)
    print(get_target_phase(State.read_json_file(Path(os.environ["STATE_FILE"])), pair))


@app.command("atoma-unlocked")
def atoma_unlocked(strategy_file: Path = typer.Option(STRATEGY_FILE)) -> None:
    """Exit 0 when the Safe's Atoma shares are unlocked, 3 while they are locked.

    The manager reports no deadline once the deposit epoch has passed. Any
    other exit status is a failure of this check, not a lock.
    """
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    pair = get_target_pair(load_universe(strategy_file), "atoma")
    lagoon = LagoonVault(web3, VaultSpec(42161, os.environ["VAULT_ADDRESS"]))
    end = AtomaVault(web3, VaultSpec(42161, pair.pool_address)).get_deposit_manager().get_redemption_delay_over(lagoon.safe_address)
    chain_time = web3.eth.get_block("latest")["timestamp"]
    if end is not None and calendar.timegm(end.timetuple()) > chain_time:
        print(f"Atoma shares are locked until {end} UTC")
        raise typer.Exit(3)
    print("Atoma shares are unlocked")


@app.command("atoma-unlock")
def atoma_unlock(strategy_file: Path = typer.Option(STRATEGY_FILE)) -> None:
    """Simulation only: mine past the Safe's Atoma share lock."""
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    assert is_anvil(web3), "atoma-unlock only advances an Anvil fork's clock; a live run waits for the real lock"
    pair = get_target_pair(load_universe(strategy_file), "atoma")
    lagoon = LagoonVault(web3, VaultSpec(42161, os.environ["VAULT_ADDRESS"]))
    vault = AtomaVault(web3, VaultSpec(42161, pair.pool_address))
    end = vault.get_deposit_manager().get_redemption_delay_over(lagoon.safe_address)
    mine(web3, timestamp=calendar.timegm(end.timetuple()) + 1)
    print(f"Arbitrum fork time advanced past the Atoma lock ending {end}")


@app.command()
def audit(strategy_file: Path = typer.Option(STRATEGY_FILE)) -> None:
    """Show every position and refuse unfinished execution before investor redemption."""
    state = State.read_json_file(Path(os.environ["STATE_FILE"]))
    universe = load_universe(strategy_file)
    rows = []
    for position in state.portfolio.get_all_positions():
        statuses = sorted({t.get_status().name for t in position.trades.values()})
        rows.append((position.position_id, position.pair.get_ticker(), "open" if position.is_open() else "closed", position.get_quantity(), position.get_value(), ",".join(statuses)))
    print(tabulate(rows, headers=["Position", "Pair", "State", "Quantity", "Value USD", "Trade status"]))
    print(f"Reserves: {state.portfolio.get_cash()} USDC; total equity {state.portfolio.calculate_total_equity()} USD")
    unfinished = [t for t in state.portfolio.get_all_trades() if not t.is_success() and t.get_status() not in {TradeStatus.expired, TradeStatus.planned}]
    if unfinished:
        raise RuntimeError(f"Unfinished trades: {[(t.trade_id, t.get_status().name) for t in unfinished]}")
    for target in ("lighter", "atoma", "upshift"):
        if get_target_phase(state, get_target_pair(universe, target)) != "done":
            raise RuntimeError(f"Target {target} is not complete")
    open_bridges = [p for p in state.portfolio.open_positions.values() if p.pair.is_cctp_bridge() and p.get_available_bridge_capital() > Decimal("0.01")]
    if open_bridges:
        raise RuntimeError(f"Bridge capital remains on satellites: {[(p.pair.get_ticker(), p.get_available_bridge_capital()) for p in open_bridges]}")


@app.command()
def finished() -> None:
    """Refuse completion while the investor holds Lagoon shares; print returned USDC."""
    web3 = create_multi_provider_web3(os.environ["JSON_RPC_ARBITRUM"])
    wallet = HotWallet.from_private_key(os.environ["PRIVATE_KEY"])
    vault = LagoonVault(web3, VaultSpec(42161, os.environ["VAULT_ADDRESS"]), trading_strategy_module_address=os.environ["VAULT_ADAPTER_ADDRESS"])
    shares = vault.share_token.fetch_raw_balance_of(wallet.address)
    if shares:
        raise RuntimeError(f"Investor {wallet.address} still holds {shares} raw Lagoon shares; finish lagoon-redeem")
    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    print(f"Investor {Web3.to_checksum_address(wallet.address)} Arbitrum USDC: {usdc.fetch_balance_of(wallet.address)}")


if __name__ == "__main__":
    app()
