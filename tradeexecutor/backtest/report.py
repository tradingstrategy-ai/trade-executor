"""Create Jupyter Notebook based report."""
import logging
import os.path
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode

from tradeexecutor.state.state import State
from tradingstrategy.client import BaseClient

from tradeexecutor.backtest.notebook import setup_charting_and_output
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation


logger = logging.getLogger(__name__)


class BacktestReporter:
    """Shared between host environment and IPython report notebook.

    A singleton instance used to communicate to IPython notebook.
    """

    def __init__(self, state: State):
        self.state = state

    def get_state(self) -> State:
        return self.state

    @classmethod
    def setup_host(cls, state):
        cls._singleton = BacktestReporter(state)

    @classmethod
    def setup_report(cls, parameters) -> "BacktestReporter":
        """Set-up notebook side reporting.

        - Output formatting

        - Reading data from the host instance
        """
        setup_charting_and_output()

        state_file = parameters["state_file"]
        state = State.read_json_file(Path(state_file))
        return BacktestReporter(
            state=state,
        )


def export_backtest_report(
        state: State,
        report_template: Path | None = None,
        output_notebook: Path | None = None,
) -> NotebookNode:
    """Creates the backtest visual report.

    - Opens a master template notebook

    - Injects the backtested state to this notebook by modifying
      the first cell of the notebook and writes a temporary state
      file path there

    - Runs the notebook

    - Writes the output notebook if specified

    - Writes the output HTML file if specified

    :return:
        Returns the executed notebook contents
    """

    assert isinstance(state, State), f"Expected State, got {state}"

    logger.info("Creating backtest result report for %s", state.name)

    if report_template is None:
        report_template = Path(os.path.join(os.path.dirname(__file__), "backtest_report_template.ipynb"))

    assert report_template.exists(), f"Does not exist: {report_template}"

    # Pass over the state to the notebook as JSON file dump
    with NamedTemporaryFile(suffix='.json', prefix=os.path.basename(__file__)) as state_temp:
        state_path = Path(state_temp.name).absolute()

        state.write_json_file(state_path)

        # https://nbconvert.readthedocs.io/en/latest/execute_api.html
        with open(report_template) as f:
            nb = nbformat.read(f, as_version=4)

        # Replace the first cell that allows us to pass parameters
        # See
        # - https://github.com/nteract/papermill/blob/main/papermill/parameterize.py
        # - https://github.com/takluyver/nbparameterise/blob/master/nbparameterise/code.py
        # for inspiration
        cell = nb.cells[0]
        assert cell.cell_type == "code", f"Assumed first cell is parameter cell, got {cell}"
        assert "parameters =" in cell.source, f"Did not see parameters = definition in the cell source: {cell.source}"
        cell.source = f"""parameters = {{"state_file": "{state_path}"}} """

        # Run the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': '.'}})

        if output_notebook is not None:
            with open(output_notebook, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

        return nb



def run_backtest_and_report(
    mod: StrategyModuleInformation,
    client: BaseClient,
):
    pass