"""Print out the trading pair data."""
import enum
import logging
from pathlib import Path
from typing import Optional

from tabulate import tabulate
from typer import Option

from tradingstrategy.client import Client
from tradingstrategy.transport.token_cache import read_token_cache, calculate_token_cache_summary, display_token_metadata
from . import shared_options
from .app import app
from .shared_options import required_option
from ..bootstrap import prepare_cache, prepare_executor_id

logger = logging.getLogger(__name__)


class PurgeType(enum.Enum):
    none = "none"
    all = "all"


class PrintTokenOption(enum.Enum):
    none = "none"
    all = "all"


@app.command()
def trading_pair(
    id: str = shared_options.id,
    strategy_file = Option(None, envvar="STRATEGY_FILE", help="Python trading strategy module to use for running the strategy"),
    cache_path: Optional[Path] = shared_options.cache_path,
    purge: PurgeType = Option("none", envvar="PURGE_TYPE", help="Which cache data should be purged"),
    unit_testing: bool = shared_options.unit_testing,
    trading_strategy_api_key: str = required_option(shared_options.trading_strategy_api_key),
):
    """Display and update trading pair data.

    Displays information regarding a particular trading pair. Allow force refresh the caches.
    """

    # Drive id and cache path from the strategy file
    if id is not None or strategy_file is not None:
        print(f"Deriving cache path from strategy id: {id} and strategy module: {strategy_file}")
        id = prepare_executor_id(id, strategy_file)
        cache_path = prepare_cache(id, cache_path, unit_testing)
        print(f"Cache path is: {cache_path}")
    else:
        # Use default ~/.tradingstrategy cache path
        pass

    client = Client.create_live_client(
        api_key=trading_strategy_api_key,
        cache_path=cache_path,  # Set via environment variable for unit testing
        settings_path=None,  # No interactive settings file with live execution
    )



    cached_entries = list(read_token_cache(client.transport))

    summary = calculate_token_cache_summary(cached_entries)

    table = [[key, value] for key, value in summary.items()]

    path = client.transport.get_abs_cache_path() / 'token-metadata'
    print(f"Cache path: {path}")
    print("Token metadata cache contents:")
    print(tabulate(table, headers=["Key", "Value"], tablefmt="fancy_grid"))

    match purge:
        case PurgeType.none:
            purge_entries = []
        case PurgeType.missing_tokensniffer_data:
            purge_entries = [e for e in cached_entries if not e.has_tokensniffer_data()]
        case PurgeType.all:
            purge_entries = cached_entries
        case _:
            raise NotImplementedError(f"PurgeType {purge} not implemented")

    if print_option == PrintTokenOption.all:
        # Print tokens
        if len(cached_entries) == 0:
            print("No data")
        else:
            print("Cache contents:")
        data = display_token_metadata(cached_entries)
        print(tabulate(data, headers="keys", tablefmt="fancy_grid"))

    if purge_entries:
        if not unit_testing:
            resp = input(f"Do you want to purge {len(purge_entries)} tokens from cache? [y/N]")
        else:
            resp = "y"
        if resp == "y":
            for e in purge_entries:
                print(f"Deleting {e.path}")
                e.purge()
            print(f"Purged {len(purge_entries)} tokens from cache")
    else:
        print("Nothing to purge")
