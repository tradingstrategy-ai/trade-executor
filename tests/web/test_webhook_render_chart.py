"""Rendering server-side charts."""

import datetime
import tempfile
from pathlib import Path
from queue import Queue

import numpy as np
import pandas as pd
import pytest
import requests
from eth_defi.utils import find_free_port
import plotly.graph_objects as go

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.webhook.server import create_webhook_server


@pytest.fixture()
def store() -> JSONFileStore:
    """Dummy state and store for the tests."""
    portfolio = Portfolio()
    state = State(portfolio=portfolio)
    store = JSONFileStore("/tmp/webhook-test.json")
    store.sync(state)
    return store


@pytest.fixture()
def chart_registry() -> ChartRegistry:
    """Test functiosn exposed for rendering."""
    def chart_func(chart_input: ChartInput):
        """Render sine wave as a Plotly figure."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Sine Wave'))
        fig.update_layout(title='Sine Wave', xaxis_title='x', yaxis_title='sin(x)')
        return fig

    def table_func(chart_input: ChartInput):
        """Render sine wave as DataFrame table."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        return pd.DataFrame({
            "x": x,
            "y": y,
        })

    chart_registry = ChartRegistry()
    chart_registry.register(chart_func, ChartKind.indicator_all_pairs)
    chart_registry.register(table_func, ChartKind.indicator_all_pairs)
    return chart_registry


@pytest.fixture()
def server_url(store, chart_registry):
    execution_state = RunState()
    execution_state.source_code = "Foobar"
    execution_state.visualisation.small_image = b"1"
    execution_state.visualisation.large_image = b"2"
    execution_state.visualisation.small_image_dark = b"3"
    execution_state.visualisation.large_image_dark = b"4"
    execution_state.version.tag = "v1"
    execution_state.version.commit_message = "Foobar"
    execution_state.read_only_state_copy = store.load()
    execution_state.chart_registry = chart_registry

    queue = Queue()

    metadata = Metadata(
        "Foobar",
        "Short desc",
        "Long desc",
        None,
        datetime.datetime.utcnow(),
        True,
        badges=Metadata.parse_badges_configuration("polygon, metamask, eth, usdc"),
        tags={StrategyTag.beta},
        fees=dict(
            management_fee=0.00,
            trading_strategy_protocol_fee=0.02,
            strategy_developer_fee=0.05,
        )
    )

    # Inject some fake files for the backtest content
    notebook_result = Path(tempfile.mkdtemp()) / 'test_cli_backtest.ipynb'
    notebook_result.open("wt").write("Foo")
    html_result = Path(tempfile.mkdtemp()) / 'test_cli_backtest.html'
    html_result.open("wt").write("Bar")
    metadata.backtest_notebook = notebook_result
    metadata.backtest_html = html_result

    port = find_free_port(20_000, 40_000, 20)
    server = create_webhook_server("127.0.0.1", port, "test", "test", queue, store, metadata, execution_state)
    server_url = f"http://test:test@127.0.0.1:{port}"
    yield server_url
    server.shutdown()


def test_web_render_figure(logger, server_url):
    """Render PNG and HTML output on the server-side for strategy charts"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry/render",
        params={"chart_id": "chart_func"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "image/png", f"Got: {resp.text}"
    assert int(resp.headers["content-length"]) > 100


def test_web_render_table(logger, server_url):
    """Render PNG and HTML output on the server-side for strategy charts"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry/render",
        params={"chart_id": "table_func"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "text/html; charset=UTF-8", f"Got: {resp.text}"
    assert int(resp.headers["content-length"]) > 100
