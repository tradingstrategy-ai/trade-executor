"""Render Lighter account share-price charts for manual inspection.

Replays every timeline of
:py:mod:`tradeexecutor.testing.lighter_share_price_timelines` into one state,
a master account and one sub-account per timeline, then draws each account's
internal share price, shares, value and USD profit to PNG files. The unit
tests assert the same timelines' figures; this script is for what asserts
miss: a spike, a gap, shares moving on a valuation, a series that does not
restart.

No chain, network or secrets are needed:

.. code-block:: shell

    poetry run python scripts/lagoon/manual-lighter-share-price-charts.py

Environment variables, all optional:

- ``OUTPUT_DIR``: where the PNG files go, default ``/tmp/lighter-share-price-charts``
- ``OPEN_BROWSER``: ``true`` also opens every chart in a browser

What to look for in the charts:

- the share price is flat across every deposit, withdrawal, allocation and
  transfer marker, except a fee step down at a ``Transfer out``;
- the share price moves only at ``Valuation`` markers;
- shares step only at flow markers, never at a valuation;
- the USD profit moves only at valuations and fees, and ends at the expected
  value printed in the table;
- ``total-loss-and-refund`` drops to zero and restarts at 1.0 on the next
  deposit;
- ``write-off-and-close`` ends at the close, with the share price at zero
  from the write-off while the table shows a 0% closed return;
- every chart starts at the account's first flow or valuation, with no
  negative shares and no gaps.

The script stops before drawing if an account breaks a flow-neutrality
invariant, naming the event and the figures.

See ``docs/plans/lighter-subaccount-flow-neutral-profit.md``.
"""

import logging
import os
import sys
from pathlib import Path

from tabulate import tabulate

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.exchange_account import all_exchange_account_share_prices, exchange_account_share_price
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.testing.lighter_share_price_timelines import TIMELINES, TimelineInvariantError, TimelineReplay
from tradeexecutor.visual.image_output import open_plotly_figure_in_browser, render_plotly_figure_as_image_file

logger = logging.getLogger("manual-lighter-share-price-charts")

#: Size of the rendered PNG files, in pixels.
CHART_WIDTH = 1200
CHART_HEIGHT = 900


def main() -> int:
    """Replay the timelines, check them, and render the charts.

    :return:
        Process exit code: 0 on success, 1 if an invariant or expectation failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The replay logs every trade and statistics update, and Kaleido every
    # browser step, at INFO.
    for noisy in ("tradeexecutor", "kaleido", "choreographer", "logistro"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/tmp/lighter-share-price-charts"))
    open_browser = os.environ.get("OPEN_BROWSER", "false").lower() == "true"
    output_dir.mkdir(parents=True, exist_ok=True)

    accounts = [account for timeline in TIMELINES for account in timeline.accounts]
    replay = TimelineReplay(accounts)
    logger.info("Replaying %d timelines over %d accounts", len(TIMELINES), len(replay.positions))
    try:
        for timeline in TIMELINES:
            replay.replay(timeline)
            replay.check_expectations(timeline)
    except TimelineInvariantError as error:
        logger.error("Invariant failed, no charts drawn: %s", error)
        return 1

    rows = []
    for timeline in TIMELINES:
        for account, expected in timeline.expected.items():
            position = replay.positions[account]
            state = position.share_price_state
            rows.append((
                timeline.timeline_id,
                account,
                "closed" if position.is_closed() else "open",
                f"{position.get_total_profit_percent():.2%}",
                f"{expected.return_pct:.2%}",
                f"{position.get_total_profit_usd():.4f}",
                f"{expected.profit_usd:.4f}",
                f"{state.total_supply:.4f}",
                f"{state.total_withdrawn:.4f}",
            ))
    logger.info("Final figures:\n%s", tabulate(rows, headers=["Timeline", "Account", "Status", "Return", "Expected", "Profit USD", "Expected", "Shares", "Withdrawn"], tablefmt="simple"))

    written = []
    for account, position in replay.positions.items():
        figure = exchange_account_share_price(ChartInput(execution_context=unit_test_execution_context, state=replay.state, pairs=[position.pair]), height=CHART_HEIGHT, width=CHART_WIDTH)
        path = output_dir / f"{account}.png"
        path.write_bytes(render_plotly_figure_as_image_file(figure, width=CHART_WIDTH, height=CHART_HEIGHT))
        written.append(path)
        if open_browser:
            open_plotly_figure_in_browser(figure, height=CHART_HEIGHT, width=CHART_WIDTH)
    overview = all_exchange_account_share_prices(ChartInput(execution_context=unit_test_execution_context, state=replay.state), width=CHART_WIDTH)
    overview_path = output_dir / "all-accounts.png"
    overview_path.write_bytes(render_plotly_figure_as_image_file(overview, width=CHART_WIDTH, height=700))
    written.append(overview_path)
    if open_browser:
        open_plotly_figure_in_browser(overview, height=700, width=CHART_WIDTH)

    logger.info("Wrote %d charts:\n%s", len(written), "\n".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
