"""API function entrypoints."""

import os
import logging
import time

from pathlib import Path
from typing import cast
from urllib.parse import urljoin

from pyramid.request import Request
from pyramid.response import Response, FileResponse
from pyramid.view import view_config

from tradeexecutor.cli.log import get_ring_buffer_handler
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.validator import validate_nested_state_dict
from tradeexecutor.strategy.chart.definition import ChartParameters, ChartInput
from tradeexecutor.strategy.chart.renderer import render_chart
from tradeexecutor.strategy.execution_context import web_server_execution_context
from tradeexecutor.strategy.summary import StrategySummary
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.visual.web_chart import WebChartType, render_web_chart, WebChartSource
from tradeexecutor.webhook.error import exception_response


logger = logging.getLogger(__name__)


@view_config(route_name='home', permission='view')
def web_home(request: Request):
    """/ endpoint.

    The homepage displays plain text version banner.
    """
    url = request.application_url
    run_state: RunState = request.registry["run_state"]
    # https://angrybirds.fandom.com/wiki/The_Flock
    return Response(f'Trade Executor daemon of Trading Strategy\nRunning strategy {run_state.executor_id}, version {run_state.version.tag}, our URL is {url}\nFor more information and questions, come to talk to us https://tradingstrategy.ai/community', content_type="text/plain")


@view_config(route_name='web_ping', renderer='json', permission='view')
def web_ping(request: Request):
    """/ping endpoint

    Unauthenticated endpoint to check the serverPlain is up.
    """
    return {"ping": "pong"}


@view_config(route_name='web_metadata', renderer='json', permission='view')
def web_metadata(request: Request):
    """/metadata endpoint

    Executor metadata.
    """
    metadata: Metadata = request.registry["metadata"]
    run_state: RunState = request.registry["run_state"]

    _icon_url = urljoin(request.application_url, "icon")
    icon_url = _icon_url.replace("http://", "https://")

    # Retrofitted with the running flag,
    # not really a nice API design.
    # Do not mutate a global state in place/
    summary = StrategySummary(
        name=metadata.name,
        short_description=metadata.short_description,
        long_description=metadata.long_description,
        icon_url=icon_url,
        started_at=time.mktime(metadata.started_at.timetuple()),
        executor_running=run_state.executor_running,
        summary_statistics=run_state.summary_statistics,
        on_chain_data=metadata.on_chain_data,
        frozen_positions=run_state.frozen_positions,
        error_message=run_state.exception.get("exception_message") if run_state.exception else None,
        backtest_available=metadata.has_backtest_data(),
        crashed_at=run_state.crashed_at,
        badges=metadata.badges,
        tags=metadata.tags,
        fees=metadata.fees,
    )

    # Catch NaN's and other data JavaScript cannot eat
    data = summary.to_dict()
    validate_nested_state_dict(data)

    r = Response(content_type="application/json")
    r.text = summary.to_json(allow_nan=False)
    return r


@view_config(route_name='web_notify', renderer='json', permission='view')
def web_notify(request: Request):
    """Notify the strategy executor about the availability of new data."""
    # TODO
    return {"status": "TODO"}


@view_config(route_name='web_state', renderer='json', permission='view')
def web_state(request: Request):
    """/state endpoint.

    Serve the latest full state of the strategy.

    Returns compressed (brotli) version if client supports it and it's available.

    :return 404:
        If the state has not been yet created
    """

    store: JSONFileStore = request.registry["store"]

    if not store.path.exists():
        logger.warning(f"Someone is eager to access the server. IP:{request.client_addr}, user agent:{request.user_agent}")
        return exception_response(404, detail="Status file not yet created")

    # Ensure "zero copy" WSGI file serving
    assert "wsgi.file_wrapper" in request.environ, "We need wsgi.file_wrapper or we will be too slow"

    # Check if we can serve Brotli compressed version
    accepts_br = "br" in request.headers.get("Accept-Encoding", "")
    br_path = Path(f"{store.path}.br")

    if (accepts_br and
        br_path.exists() and
        br_path.stat().st_mtime >= store.path.stat().st_mtime):
        # Serve compressed version if client supports it and it's available
        r = FileResponse(content_type="application/json", request=request, path=br_path)
        r.headers["Content-Encoding"] = "br"
    else:
        # Fallback to uncompressed
        r = FileResponse(content_type="application/json", request=request, path=store.path)

    # Ensure caches store separate entries for different Accept-Encoding values
    r.headers["Vary"] = "Accept-Encoding"

    return r

@view_config(route_name='web_status', renderer='json', permission='view')
def web_status(request: Request):
    """/status endpoint.

    Return if the trade-executor is still alive or the exception that crashed it.

    See :py:class:`tradeexecutor.strategy.execution_state.ExecutionState` for the return dta.
    """
    run_state: RunState = request.registry["run_state"]

    r = Response(content_type="application/json")
    try:
        r.text = run_state.make_exportable_copy().to_json()
    except Exception as e:
        logger.error("web_status(): could not export run-time state", exc_info=e)
        return exception_response(501, detail=f"Run-time run_state corrupted")

    return r


@view_config(route_name='web_logs', renderer='json', permission='view')
def web_logs(request: Request):
    """/logs endpoint.

    Return if the trade-executor is still alive or the exception that crashed it.

    See :py:class:`tradeexecutor.strategy.execution_state.ExecutionState` for the return dta.
    """
    ring_buffer_handler = get_ring_buffer_handler()
    assert ring_buffer_handler is not None, "In-memory logging not initialised"
    logs = ring_buffer_handler.export()
    return logs


@view_config(route_name='web_source', permission='view')
def web_source(request: Request):
    """/source endpoint.

    Return the source code of the strategy as plain text.
    """
    execution_state: RunState = request.registry["run_state"]
    r = Response(content_type="text/plain")
    r.text = execution_state.source_code or ""
    return r


@view_config(route_name='web_visualisation', permission='view')
def web_visulisation(request: Request):
    """/visualisation endpoint.

    Return strategy images.

    See :py:func:`tradeexecutor.strategy.pandas_trader.report_strategy_thinking` for more information.
    """
    execution_state: RunState = request.registry["run_state"]

    type = request.params.get("type", "small")
    theme = request.params.get("theme", "light")

    logger.info("Reading visualisation image, last updated %s", execution_state.visualisation.last_refreshed_at)

    if type == "small":

        if theme == "light":
            data = execution_state.visualisation.small_image
        else:
            data = execution_state.visualisation.small_image_dark

        if not data:
            return exception_response(501, detail=f"Image data not available. It will be generated on the first strategy cycle.")

        r = Response(content_type="image/svg+xml")
        r.body = data
        return r
    elif type =="large":

        if theme == "light":
            data = execution_state.visualisation.large_image
        else:
            data = execution_state.visualisation.large_image_dark

        if not data:
            return exception_response(501, detail=f"Image data not available. It will be generated on the first strategy cycle.")

        r = Response(content_type="image/svg+xml")
        r.body = data
        return r
    else:
        # Use 501 Not implemented error code
        return exception_response(501, detail=f"Not implemented. Unknown type {type}")


@view_config(route_name='web_file', permission='view')
def web_file(request: Request):
    """/file endpoint.

    Serve some trading strategy related files.
    """
    metadata = cast(Metadata, request.registry["metadata"])

    type = request.params.get("type")
    match type:
        case "notebook":
            path = metadata.backtest_notebook
            # https://docs.jupyter.org/en/latest/reference/mimetype.html
            content_type = "application/x-ipynb+json"
        case "html":
            path = metadata.backtest_html
            content_type = "text/html"
        case _:
            return exception_response(501, detail=f"Not implemented. Unknown type {type}")

    if not path or not path.exists():
        return exception_response(404, detail=f"Backtest data not available for {type}")

    r = FileResponse(path.as_posix(), content_type=content_type)
    return r


@view_config(route_name='web_chart', permission='view', renderer='json')
def web_chart(request: Request):
    """/chart endpoint.

    Return chart data for the frontend rendering.

    .. note ::

        Legacy. Does not do `ChartRegistry` charts.

    Unlike other endpoints, this endpoint does processing, albeit light.
    Under wrong circumstances
    """

    run_state: RunState = request.registry["run_state"]

    type_str = request.params.get("type")
    source_str = request.params.get("source")

    try:
        type = WebChartType(type_str)
    except:
        return exception_response(501, detail=f"Not implemented. Unknown chart type {type_str}")

    try:
        source = WebChartSource(source_str)
    except:
        return exception_response(501, detail=f"Not implemented. Unknown source {source_str}")

    if source == WebChartSource.live_trading:
        # Use read-only state copy to calculate charts
        state = run_state.read_only_state_copy
        assert state is not None, "We asked for teh RunState.read_only_state_copy but it was not set in this point"
    else:
        metadata = cast(Metadata, request.registry["metadata"])
        state = metadata.backtested_state
        if not state or state.is_empty():
            return exception_response(404, detail=f"Backtest data not available")

    data = render_web_chart(state, type, source)
    r = Response(content_type="application/json")
    r.text = data.to_json()
    return r


@view_config(route_name='web_icon', permission='view')
def web_icon(request: Request):
    """Render the icon of the executor."""
    path = Path(os.path.join(os.path.dirname(__file__), "default_logo.png"))
    assert path.exists(), f"Does not exist {path}"
    r = FileResponse(path.as_posix(), content_type="image/png")
    return r


@view_config(route_name='web_chart_render', permission='view')
def web_chart_render(request: Request):
    """/chart-registry/render endpoint.

    - Return chart data for the frontend rendering.
    - Can generate text/html, text/plain, image/png or image/svg response
    - Sets `X-Error` header if there was an error rendering the chart,
      then having traceback as the payload

    :return:
        HTML or PNG or SVG image data for the chart.
    """
    run_state: RunState = request.registry["run_state"]
    chart_registry = run_state.chart_registry
    chart_id = request.params.get("chart_id")
    pair_ids = request.params.get("pair_ids")
    format = request.params.get("format", "png").lower()

    pairs = []

    strategy_input_indicators = run_state.latest_indicators
    if strategy_input_indicators is not None:
        strategy_universe = strategy_input_indicators.strategy_universe
    else:
        # We don't necessarily care about this in test paths
        strategy_universe = None

    if pair_ids:
        assert strategy_universe is not None, "pair_ids provided but strategy_universe is None, cannot parse pair ids"
        split = pair_ids.split(",")
        for pair_id in split:
            try:
                pairs.append(strategy_universe.get_pair_by_id(int(pair_id)))
            except ValueError:
                return exception_response(501, detail=f"Bad pair id: {pair_id}")

    chart_response = render_chart(
        registry=chart_registry,
        chart_id=chart_id,
        parameters=ChartParameters(
            format=format,
            width=1200,
            height=800,
        ),
        input=ChartInput(
            execution_context=web_server_execution_context,
            strategy_input_indicators=run_state.latest_indicators,
            state=run_state.read_only_state_copy,
            pairs=pairs,
        ),
    )

    # We got an exception rendering the chart
    if chart_response.error:
        r = Response(content_type="text/plain")
        r.headers["X-Error"] = "true"
        r.body = chart_response.error.encode("utf-8")
    else:
        r = Response(content_type=chart_response.content_type)
        r.body = chart_response.data
    return r