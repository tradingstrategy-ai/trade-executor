"""Command-line entry point for the daemon build on the top of Typer."""
from .commands import enzyme_deploy_vault
from .commands.app import app
from .commands.backtest import backtest
from .commands.check_accounts import check_accounts
from .commands.check_universe import check_universe
from .commands.check_wallet import check_wallet
from .commands.close_all import close_all
from .commands.console import console
from .commands.correct_accounts import correct_accounts
from .commands.enzyme_asset_list import enzyme_asset_list
from .commands.export import export
from .commands.hello import hello
from .commands.reset import reset
from .commands.reset_deposits import reset_deposits
from .commands.show_positions import show_positions
from .commands.start import start
from .commands.perform_test_trade import perform_test_trade
from .commands.version import version
from .commands.repair import repair
from .commands.retry import retry
from .commands.visualise import visualise
from .commands.init import init
from .commands.webapi import webapi

# Dummy export commands even though they are already registered
# to make the linter happy
__all__ = [
    app, check_wallet, check_universe, hello, start, perform_test_trade, 
    version, repair, console, init, reset, enzyme_asset_list, enzyme_deploy_vault,
    close_all, show_positions, backtest, correct_accounts, check_accounts, 
    reset_deposits, export, retry, visualise, webapi
]
