"""Inject iframe CSS/JS into an external HTML file for use as a backtest report.

.. code-block:: console

    trade-executor prepare-report /path/to/external-report.html --id my-strategy

"""

import logging
from pathlib import Path
from typing import Optional

import typer

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id

logger = logging.getLogger(__name__)


@app.command()
def prepare_report(
    input_html: Path = typer.Argument(..., help="Path to the externally generated HTML file"),
    id: str = shared_options.id,
    strategy_file: Path = typer.Option(None, envvar="STRATEGY_FILE", help="Strategy file used to derive the executor id if --id is not given"),
    log_level: str = shared_options.log_level,
):
    """Inject iframe CSS/JS into an external HTML file and place it as the backtest report.

    Reads the input HTML, injects the standard CSS/JS for iframe embedding,
    and writes it to state/{id}-backtest.html so the executor can serve it.
    """
    global logger

    from tradeexecutor.cli.log import setup_logging
    from tradeexecutor.backtest.tearsheet import prepare_html_for_iframe

    id = prepare_executor_id(id, strategy_file)

    if not log_level:
        log_level = logging.WARNING
    logger = setup_logging(log_level)

    if not input_html.exists():
        raise typer.BadParameter(f"Input HTML file does not exist: {input_html}")

    output_html = Path(f"state/{id}-backtest.html")

    prepare_html_for_iframe(input_html, output_html)

    typer.echo(f"Wrote iframe-ready backtest report to {output_html}")
