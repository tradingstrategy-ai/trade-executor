"""Command-line entry point for the daemon build on the top of Typer."""
from .commands import console, enzyme_deploy_vault
from .commands.app import app
from .commands.check_universe import check_universe
from .commands.check_wallet import check_wallet
from .commands.enzyme_asset_list import enzyme_asset_list
from .commands.hello import hello
from .commands.start import start
from .commands.perform_test_trade import perform_test_trade
from .commands.version import version
from .commands.repair import repair
from .commands.init import init


# Dummy export commands even though they are already registered
# to make the linter happy
__all__ = [app, check_wallet, check_universe, hello, start, perform_test_trade, version, repair, console, init, enzyme_asset_list, enzyme_deploy_vault]











