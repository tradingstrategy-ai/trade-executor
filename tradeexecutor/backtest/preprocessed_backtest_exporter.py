"""Prebuilt backtest dataset generation entry point.

To export / update all exported data:

.. code-block:: shell

    python tradeexecutor/backtest/preprocessed_backtest_exporter.py ~/exported

"""

import logging
import os
import sys
from pathlib import Path

from tradeexecutor.backtest.preprocessed_backtest import PREPACKAGED_SETS, prepare_dataset
from tradeexecutor.cli.log import setup_logging
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


def export_all_main():
    """Export all preprocessed backtest sets.

    - Main entry point
    """

    setup_logging()

    client = Client.create_live_client(api_key=os.environ["TRADING_STRATEGY_API_KEY"])
    output_path = Path(sys.argv[1])

    if len(sys.argv) == 3:
        slug = sys.argv[2]
    else:
        slug = None

    assert output_path.exists(), f"{output_path} does not exist"
    assert output_path.is_dir(), f"{output_path} is not a directory"
    for ds in PREPACKAGED_SETS:

        # filter by slug
        if slug:
            if ds.slug != slug:
                continue

        prepare_dataset(
            client=client,
            dataset=ds,
            output_folder=output_path,
        )

    logger.info("All done")


if __name__ == "__main__":
    export_all_main()