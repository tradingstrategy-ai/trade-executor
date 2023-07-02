"""Create Jupyter Notebook based report."""
import os.path
from zipfile import Path

from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import nbformat
from tradingstrategy.client import BaseClient

from tradeexecutor.strategy.strategy_module import StrategyModuleInformation


def create_backtest_report(
        mod: StrategyModuleInformation,
        client: BaseClient,
        report_template: Path | None = None,
        output_notebook: Path | None = None,
        output_state: Path | None = None,
):
    """Runs a strategy test using a notebook and generates a report."""

    if report_template is None:
        report_template = os.path.dirname(__file__) / "backtest_report_template.ipynb"

    # https://nbconvert.readthedocs.io/en/latest/execute_api.html
    with open(report_template) as f:
        nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')