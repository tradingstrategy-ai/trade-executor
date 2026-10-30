"""Render the existing position charts for Lighter accounts, for manual inspection.

Replays every timeline of
:py:mod:`tradeexecutor.testing.lighter_share_price_timelines` into one state,
a master account and one sub-account per timeline, then draws every account
with the chart a strategy uses for a single position,
:py:func:`tradeexecutor.strategy.chart.standard.vault.vault_position_timeline`,
and all of them with
:py:func:`tradeexecutor.strategy.chart.standard.weight.equity_curve_by_asset`,
to PNG files. For an exchange account the position chart's price is the
internal share price, see
:py:func:`tradeexecutor.visual.position.calculate_position_timeline`. The unit
tests assert the same timelines' figures; this script is for what asserts
miss: a spike, a gap, shares moving on a valuation, a series that does not
restart.

No chain, network or secrets are needed:

.. code-block:: shell

    poetry run python scripts/lagoon/manual-lighter-share-price-charts.py

Environment variables, all optional:

- ``OUTPUT_DIR``: where the PNG files go, default ``/tmp/lighter-share-price-charts``
- ``OPEN_BROWSER``: ``true`` also opens every chart in a browser

What to look for in the position charts:

- the internal share price is flat at every trade marker, which marks shares
  minted (▲, a deposit, allocation or transfer in) or burned (▼, a
  withdrawal or transfer out), except a fee step down on a transfer out;
- the share price moves only between markers, at valuations;
- the position value moves at every flow and valuation;
- the PnL USD panel moves only at valuations and fees, and ends at the
  expected value printed in the table;
- ``total-loss-and-refund`` drops to zero and restarts at 1.0 on the next
  deposit;
- ``write-off-and-close`` ends at the close, with the share price at zero
  from the write-off while the PnL USD panel ends at +0.5 and the table
  shows a +5% closed return;
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

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.vault import vault_position_timeline
from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_asset
from tradeexecutor.strategy.execution_context import unit_test_trading_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
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

    # A live execution context: the charts take their time range from the
    # statistics, which the replay records after every event.
    indicators = StrategyInputIndicators(strategy_universe=replay.create_strategy_universe(), available_indicators=IndicatorSet(), indicator_results={})

    def chart_input(pairs: list[TradingPairIdentifier] | None = None) -> ChartInput:
        return ChartInput(execution_context=unit_test_trading_execution_context, state=replay.state, strategy_input_indicators=indicators, pairs=pairs)

    written = []
    for account, position in replay.positions.items():
        figure, _df = vault_position_timeline(chart_input([position.pair]), height=CHART_HEIGHT, width=CHART_WIDTH)
        path = output_dir / f"{account}.png"
        path.write_bytes(render_plotly_figure_as_image_file(figure, width=CHART_WIDTH, height=CHART_HEIGHT))
        written.append(path)
        if open_browser:
            open_plotly_figure_in_browser(figure, height=CHART_HEIGHT, width=CHART_WIDTH)
    overview = equity_curve_by_asset(chart_input())
    overview_path = output_dir / "all-accounts.png"
    overview_path.write_bytes(render_plotly_figure_as_image_file(overview, width=CHART_WIDTH, height=700))
    written.append(overview_path)
    if open_browser:
        open_plotly_figure_in_browser(overview, height=700, width=CHART_WIDTH)

    logger.info("Wrote %d charts:\n%s", len(written), "\n".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
