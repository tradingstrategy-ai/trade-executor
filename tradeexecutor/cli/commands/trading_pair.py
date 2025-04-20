"""Print out the trading pair data."""
import datetime
import enum
import logging
from pathlib import Path
from typing import Optional

from tabulate import tabulate
from typer import Option

from . import shared_options
from .app import app
from .shared_options import required_option
from ..bootstrap import prepare_cache, prepare_executor_id
from ..log import setup_logging
from ..universe import load_universe
from ...state.identifier import TradingPairIdentifier
from ...strategy.bootstrap import import_strategy_file
from ...strategy.execution_context import console_command_execution_context
from ...strategy.trading_strategy_universe import TradingStrategyUniverse


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
    strategy_file: Path = shared_options.strategy_file,
    cache_path: Optional[Path] = shared_options.cache_path,
    max_data_delay_minutes: int = shared_options.max_data_delay_minutes,
    log_level: str = shared_options.log_level,
    max_workers: int | None = shared_options.max_workers,
    purge: PurgeType = Option("none", envvar="PURGE_TYPE", help="Which cache data should be purged"),
    unit_testing: bool = shared_options.unit_testing,
    trading_strategy_api_key: str = required_option(shared_options.trading_strategy_api_key),
    token_address: str = Option(..., envvar="TOKEN_ADDRESS", help="ERC-20 address of base token of a trading pair"),
):
    """Display and update trading pair data.

    Displays information regarding a particular trading pair. Allow force refresh the caches.
    """

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)
    if log_level != "disabled":
        assert logger.level <= logging.INFO, "Log level must be at least INFO to get output from this command"

    logger.info("Loading strategy file %s", strategy_file)

    strategy_factory = import_strategy_file(strategy_file)

    cache_path = prepare_cache(id, cache_path)

    assert trading_strategy_api_key, "TRADING_STRATEGY_API_KEY missing"

    execution_context = console_command_execution_context

    universe_init = load_universe(
        trading_strategy_api_key=trading_strategy_api_key,
        cache_path=cache_path,
        max_data_delay_minutes=max_data_delay_minutes,
        strategy_factory=strategy_factory,
        execution_context=execution_context,
    )

    # Deconstruct strategy input
    universe_model = universe_init.universe_model
    universe_options = universe_init.universe_options

    ts = datetime.datetime.utcnow()
    logger.info("Performing universe data check for timestamp %s", ts)
    strategy_universe = universe_model.construct_universe(ts, execution_context.mode, universe_options)

    assert isinstance(strategy_universe, TradingStrategyUniverse)

    trading_pairs: list[TradingPairIdentifier] = []

    token_address = token_address.lower()
    if token_address:
        for pair in strategy_universe.iterate_pairs():
            if pair.base.address == token_address:
                trading_pairs.append(pair)

    logger.info(f"We have total {strategy_universe.get_pair_count()} trading pairs")
    logger.info(f"Matched {len(trading_pairs)} pairs")
    for idx, pair in enumerate(trading_pairs):
        logger.info(f"Pair #{idx + 1}")
        info = pair.get_diagnostics_info(extended=True)

        table_data = [[k, v] for k, v in info.items()]
        logger.info("\n%s", tabulate(table_data, headers=["Property", "Value"], tablefmt="fancy_grid"))


    # if print_option == PrintTokenOption.all:
    #     # Print tokens
    #     if len(cached_entries) == 0:
    #         print("No data")
    #     else:
    #         print("Cache contents:")
    #     data = display_token_metadata(cached_entries)
    #     print(tabulate(data, headers="keys", tablefmt="fancy_grid"))
    #
    # if purge_entries:
    #     if not unit_testing:
    #         resp = input(f"Do you want to purge {len(purge_entries)} tokens from cache? [y/N]")
    #     else:
    #         resp = "y"
    #     if resp == "y":
    #         for e in purge_entries:
    #             print(f"Deleting {e.path}")
    #             e.purge()
    #         print(f"Purged {len(purge_entries)} tokens from cache")
    # else:
    #     print("Nothing to purge")
