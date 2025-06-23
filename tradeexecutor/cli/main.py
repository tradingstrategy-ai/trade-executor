"""Command-line entry point for the daemon build on the top of Typer."""

from .commands import enzyme_deploy_vault
from .commands.app import app
from .commands.backtest import backtest
from .commands.check_accounts import check_accounts
from .commands.check_universe import check_universe
from .commands.check_wallet import check_wallet
from .commands.close_all import close_all
from .commands.close_position import close_position
from .commands.console import console
from .commands.correct_accounts import correct_accounts
from .commands.deploy_guard import deploy_guard
from .commands.enzyme_asset_list import enzyme_asset_list
from .commands.export import export
from .commands.hello import hello
from .commands.reset import reset
from .commands.reset_deposits import reset_deposits
from .commands.show_positions import show_positions
from .commands.start import start
from .commands.perform_test_trade import perform_test_trade
from .commands.token_cache import token_cache
from .commands.trading_pair import trading_pair
from .commands.version import version
from .commands.repair import repair
from .commands.retry import retry
from .commands.visualise import visualise
from .commands.init import init
from .commands.webapi import webapi
from .commands.check_position_triggers import check_position_triggers
from .commands.send_log_message import send_log_message
from .commands.lagoon_deploy_vault import lagoon_deploy_vault
from .commands.show_valuation import show_valuation
from .commands.blacklist import blacklist
from .commands.prune import prune_state

# Dummy export commands even though they are already registered
# to make the linter happy
__all__ = [
    app, backtest, blacklist, check_accounts, check_position_triggers,
    check_universe, check_wallet, close_all, close_position, console,
    correct_accounts, deploy_guard, enzyme_asset_list, enzyme_deploy_vault,
    export, hello, init, lagoon_deploy_vault, perform_test_trade, prune_state,
    repair, reset, reset_deposits, retry, send_log_message, show_positions,
    show_valuation, start, token_cache, trading_pair, version, visualise,
    webapi
]
