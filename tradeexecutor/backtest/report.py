"""Create Jupyter Notebook based report.

Further reading

- `Export notebook HTML with embedded images <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/export_embedded/readme.html>`__
"""

import logging
import os.path
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import nbformat
from bs4 import BeautifulSoup
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.client import BaseClient

from tradeexecutor.backtest.notebook import setup_charting_and_output
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation


logger = logging.getLogger(__name__)



#: The default custom CSS overrides for the notebook HTML export
#:
DEFAULT_CUSTOM_CSS = """
/* trade-executor backtest report generator custom CSS */

.prompt {
    display: none !important;
}

#notebook-container {
    padding: 0;
    box-shadow: none;
}
"""


class BacktestReportRunFailed(Exception):
    """Generating a backtest report failed.

    See the wrapped :py:class:`nbclient.exceptions.CellExecutionError` for more information.
    """


class BacktestReporter:
    """Shared between host environment and IPython report notebook.

    A singleton instance used to communicate to IPython notebook.
    """

    def __init__(self, state: State, universe: TradingStrategyUniverse):
        """To write a report we need to inputs

        :param state:
            State that is the resulting trades that were made

        :param universe:
            Trading universe where the results where traded
        """
        self.state = state
        self.universe = universe

    def get_state(self) -> State:
        return self.state

    def get_universe(self) -> TradingStrategyUniverse:
        return self.universe

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
        universe_file = parameters["universe_file"]
        state = State.read_json_file(Path(state_file))
        universe = TradingStrategyUniverse.read_pickle_dangerous(Path(universe_file))
        return BacktestReporter(
            state=state,
            universe=universe,
        )


def export_backtest_report(
        state: State,
        universe: TradingStrategyUniverse,
        report_template: Path | None = None,
        output_notebook: Path | None = None,
        output_html: Path | None = None,
        show_code=False,
        custom_css: str | None=DEFAULT_CUSTOM_CSS,
) -> NotebookNode:
    """Creates the backtest visual report.

    - Opens a master template notebook

    - Injects the backtested state to this notebook by modifying
      the first cell of the notebook and writes a temporary state
      file path there

    - Runs the notebook

    - Writes the output notebook if specified

    - Writes the output HTML file if specified

    :param show_code:
        For the HTML report, should we hide the code cells.

    :param custom_css:
        CSS code to inject to the resulting HTML file to override styles.

    :return:
        Returns the executed notebook contents

    :raise BacktestReportRunFailed:
        In the case the notebook had a run-time exception and Python code could not complete.
    """

    assert isinstance(state, State), f"Expected State, got {state}"

    name = state.name
    logger.info("Creating backtest result report for %s", name)

    if report_template is None:
        report_template = Path(os.path.join(os.path.dirname(__file__), "backtest_report_template.ipynb"))

    assert report_template.exists(), f"Does not exist: {report_template}"

    # Pass over the state to the notebook as JSON file dump
    with NamedTemporaryFile(suffix='.json', prefix=os.path.basename(__file__)) as state_temp, \
        NamedTemporaryFile(suffix='.pickle', prefix=os.path.basename(__file__)) as universe_temp:

        state_path = Path(state_temp.name).absolute()
        state.write_json_file(state_path)

        universe_path = Path(universe_temp.name).absolute()
        universe.write_pickle(universe_path)

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
        cell.source = f"""parameters = {{
            "state_file": "{state_path}",
            "universe_file": "{universe_path}", 
        }} """

        # Run the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        try:
            ep.preprocess(nb, {'metadata': {'path': '.'}})
        except CellExecutionError as e:
            raise BacktestReportRunFailed(f"Could not run backtest reporter for {name}") from e

        if output_notebook is not None:
            with open(output_notebook, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

        if output_html is not None:

            html_exporter = HTMLExporter(
                template_name='classic',
                embed_images=True,
                exclude_input=show_code is False,
                exclude_input_prompt=True,
                exclude_output_prompt=True,
            )
            # Image are inlined in the output
            html_content, resources = html_exporter.from_notebook_node(nb)

            # Inject our custom css
            if custom_css is not None:
                html_content = _inject_custom_css(html_content, custom_css)

            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info("Wrote HTML report to %s, total %d bytes", output_html, len(html_content))

        return nb


def run_backtest_and_report(
    mod: StrategyModuleInformation,
    client: BaseClient,
):
    pass


def _inject_custom_css(html: str, css_code: str) -> str:
    """Injects new <style> tag to HTML code.

    Use BeautifulSoup to parse HTMl, inject new <style> tag, reassemble.

    The resulting HTML looks like:

    .. code-block:: text

        <html>
            <head>
                ...
                <style id="trade-executor-css-inject">
                    ...
    """
    assert css_code
    soup = BeautifulSoup(html)
    head = soup.head
    # Add a style tag class for better diagnostics
    tag = soup.new_tag('style', attrs={"id": "trade-executor-css-inject"}, type='text/css')
    head.append(tag)
    tag.append(css_code)
    return str(soup)