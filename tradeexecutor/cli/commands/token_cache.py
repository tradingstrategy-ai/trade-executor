"""Print out token cache status of the client."""
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
from ..bootstrap import prepare_cache

logger = logging.getLogger(__name__)


class PurgeType(enum.Enum):
    none = "none"
    all = "all"
    missing_tokensniffer_data = "missing_tokensniffer_data"


class PrintTokenOption(enum.Enum):
    none = "none"
    all = "all"


@app.command()
def token_cache(
    id: str = shared_options.id,
    cache_path: Optional[Path] = shared_options.cache_path,
    purge: PurgeType = Option("none", envvar="PURGE_TYPE", help="Which cache entries to purge"),
    print_option: PrintTokenOption = Option("none", envvar="PRINT_TOKENS", help="Which token metadata to print"),
    unit_testing: bool = shared_options.unit_testing,
    trading_strategy_api_key: str = required_option(shared_options.trading_strategy_api_key),
):
    """Display and purge token cache entries.

    - Token metadata cache contains data from TokenSniffer and CoinGecko APIs that may be stale
    """

    cache_path = prepare_cache(id, cache_path, unit_testing)

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
