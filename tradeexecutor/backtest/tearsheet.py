"""Create Jupyter Notebook based report static HTML tearsheet for a strategy.

Further reading

- `Export notebook HTML with embedded images <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/export_embedded/readme.html>`__
"""

import logging
import os.path
from pathlib import Path
from tempfile import NamedTemporaryFile

import nbformat
import pandas as pd
from bs4 import BeautifulSoup
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode

from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.notebook import setup_charting_and_output
from tradeexecutor.visual.equity_curve import calculate_daily_returns

logger = logging.getLogger(__name__)



#: The default custom CSS overrides for the notebook HTML export
#:
DEFAULT_CUSTOM_CSS = """
/* trade-executor backtest report generator custom CSS */
body {
    padding: 0;   
}

.prompt {
    display: none !important;
}

#notebook {
    padding-top: 0 !important;
}

#notebook-container {
    padding: 0 !important;
    box-shadow: none;
    width: auto;
}

.code_cell {
    padding: 0;
}
"""


# Iframe height hack
# https://stackoverflow.com/a/44547866/315168
DEFAULT_CUSTOM_JS = """
console.log("Dynamic iframe resizer loaded");

function getDocHeight(doc) {
    // stackoverflow.com/questions/1145850/
    var body = doc.body, html = doc.documentElement;
    var height = Math.max( body.scrollHeight, body.offsetHeight, 
        html.clientHeight, html.scrollHeight, html.offsetHeight );
    return height;
}

window.addEventListener("load", function(){
    if(window.self === window.top) return; // if w.self === w.top, we are not in an iframe 
    send_height_to_parent_function = function(){
        //var height = document.getElementsByTagName("html")[0].clientHeight;
        //var height= document.getElementById('wrapper').offsetHeight;
        const height = getDocHeight(document);
        console.log("Sending height as " + height + "px");
        parent.postMessage({"iframeContentHeight" : height }, "*");
    }
    // send message to parent about height updates
    send_height_to_parent_function(); //whenever the page is loaded
    window.addEventListener("resize", send_height_to_parent_function); // whenever the page is resized
    var observer = new MutationObserver(send_height_to_parent_function);           // whenever DOM changes PT1
    var config = { attributes: true, childList: true, characterData: true, subtree:true}; // PT2
    observer.observe(window.document, config);                                            // PT3 
});
"""


class BacktestReportRunFailed(Exception):
    """Generating a backtest report failed.

    See the wrapped :py:class:`nbclient.exceptions.CellExecutionError` for more information.
    """


class BacktestReporter:
    """Shared between host environment and IPython report notebook.

    - A helper class to pass data in the notebook report generation context
      using temporary files.

    - Files are written by the host system after running the backtest

    - Files are passed as absolute paths in the first notebook cell
      that is modified before the notebook is executed
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
    def setup_report(cls, parameters) -> "BacktestReporter":
        """Set-up notebook side reporting.

        - Output formatting

        - Reading data from the host instance
        """

        setup_charting_and_output()

        # By default matplotlib exports text as 2d curves to make sure
        # SVG renders correctly. However this will result to massive file sizes.
        # Here we hint matplotlib to export SVG labels as text.
        # Furthermore SVG labels do not show in the static HTML output otherwise.
        # https://stackoverflow.com/questions/34387893/output-matplotlib-figure-to-svg-with-text-as-text-not-curves
        # https://matplotlib.org/stable/users/explain/fonts.html#fonts-in-svg
        import matplotlib.pyplot as plt
        plt.rcParams['svg.fonttype'] = 'none'

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
    *,
    state_path: Path,
    universe_path: Path,
    report_template: Path | None = None,
    output_notebook: Path | None = None,
    output_html: Path | None = None,
    output_csv_daily_returns: Path | None = None,
    show_code=False,
    custom_css: str | None=DEFAULT_CUSTOM_CSS,
    custom_js: str | None=DEFAULT_CUSTOM_JS,
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

    :param custom_js:
        JS code to inject to the resulting HTML file to support iframe embedding.

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
        "state_file": "{state_path.absolute()}",
        "universe_file": "{universe_path.absolute()}", 
    }} """

    # Run the notebook
    state_size = os.path.getsize(state_path)
    universe_size = os.path.getsize(universe_path)
    logger.info(f"Starting backtest tearsheet notebook execution, state size is {state_size:,}b, universe size is {universe_size:,}b")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
    except CellExecutionError as e:
        print(e)
        raise BacktestReportRunFailed(f"Could not run backtest reporter for {name}") from e

    logger.info("Notebook executed")

    # Write ipynb file that contains output cells created in place
    if output_notebook is not None:
        with open(output_notebook, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    if output_csv_daily_returns is not None:
        returns_series = calculate_daily_returns(state)
        returns_series.index.name = 'timestamp'
        returns_series = returns_series.fillna(0)
        returns_df = pd.DataFrame({"daily_returns": returns_series})
        returns_df.to_csv(output_csv_daily_returns, index=True, float_format='%.8f')

    # Write a static HTML file based on the notebook
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
            html_content = _inject_custom_css_and_js(html_content, custom_css, custom_js)

        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info("Wrote HTML report to %s, total %d bytes", output_html, len(html_content))

    return nb


def _inject_custom_css_and_js(html: str, css_code: str, js_code: str) -> str:
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
    soup = BeautifulSoup(html, features='lxml')
    head = soup.head
    # Add a style tag class for better diagnostics
    tag = soup.new_tag('style', attrs={"id": "trade-executor-css-inject"}, type='text/css')
    head.append(tag)
    tag.append(css_code)

    tag = soup.new_tag('script', attrs={"id": "trade-executor-js-inject"})
    head.append(tag)
    tag.append(js_code)

    return str(soup)
