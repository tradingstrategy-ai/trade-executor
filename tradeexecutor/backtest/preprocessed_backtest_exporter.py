"""Prebuilt backtest dataset generation entry point.

For usage see :py:mod:`preprocessed_backtest`.
"""
import datetime
import logging
import os
import sys
from pathlib import Path

from tradeexecutor.backtest.preprocessed_backtest import PREPACKAGED_SETS, prepare_dataset, ExportFormat
from tradeexecutor.cli.log import setup_logging
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


def export_all_main():
    """Export all preprocessed backtest sets.

    - Main entry point
    """

    setup_logging()

    BACKTEST = os.environ.get("BACKTEST", "true").lower() == "true"

    client = Client.create_live_client(api_key=os.environ["TRADING_STRATEGY_API_KEY"])
    output_path = Path(sys.argv[1])

    if len(sys.argv) == 3:
        slug = sys.argv[2]
    else:
        slug = None

    reverse = False  # TODO: Hack
    if slug == "--reverse":
        reverse = True
        slug = None

    assert output_path.exists(), f"{output_path} does not exist"
    assert output_path.is_dir(), f"{output_path} is not a directory"

    started = datetime.datetime.utcnow()

    # Export newly added sets first
    if reverse:
        PREPACKAGED_SETS.reverse()

    for ds in PREPACKAGED_SETS:

        # filter by slug
        if slug:
            if ds.slug != slug:
                continue

        prepare_dataset(
            client=client,
            dataset=ds,
            output_folder=output_path,
            write_csv=False,
            write_parquet=ExportFormat.parquet in ds.formats,
            write_csv_pair_columns=ExportFormat.csv_pair_columns in ds.formats,
        )

    logger.info("All done in %s", datetime.datetime.utcnow() - started)


if __name__ == "__main__":
    export_all_main()