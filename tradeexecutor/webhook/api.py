"""API function entrypoints."""

import os
import logging
import time
from importlib.metadata import version

from pyramid.request import Request
from pyramid.response import Response, FileResponse
from pyramid.view import view_config

from tradeexecutor.cli.log import get_ring_buffer_handler
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.validator import validate_state_serialisation, validate_nested_state_dict
from tradeexecutor.strategy.summary import StrategySummary
from tradeexecutor.strategy.run_state import RunState
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
    return Response(f'Chuck the Trade Executor, running strategy {run_state.executor_id}, version {run_state.version.tag}, our URL is {url}\nFor more information see https://tradingstrategy.ai\nRemember to play Angry Birds.', content_type="text/plain")


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
    execution_state: RunState = request.registry["run_state"]

    # Retrofitted with the running flag,
    # not really a nice API design.
    # Do not mutate a global state in place/
    summary = StrategySummary(
        name=metadata.name,
        short_description=metadata.short_description,
        long_description=metadata.long_description,
        icon_url=metadata.icon_url,
        started_at=time.mktime(metadata.started_at.timetuple()),
        executor_running=execution_state.executor_running,
        summary_statistics=execution_state.summary_statistics,
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

    Serve the latest full state of the bog.

    :return 404:
        If the state has not been yet created
    """

    # Does "zero copy" WSGI file serving
    store: JSONFileStore = request.registry["store"]
    fname = store.path

    if not os.path.exists(fname):
        logger.warning("Someone is eager to access the serverPlain. IP:%s, user agent:%s", request.client_addr, request.user_agent)
        return exception_response(404, detail="Status file not yet created")

    assert 'wsgi.file_wrapper' in request.environ, "We need wsgi.file_wrapper or we will be too slow"
    r = FileResponse(content_type="application/json", request=request, path=fname)
    return r


@view_config(route_name='web_status', renderer='json', permission='view')
def web_status(request: Request):
    """/status endpoint.

    Return if the trade-executor is still alive or the exception that crashed it.

    See :py:class:`tradeexecutor.strategy.execution_state.ExecutionState` for the return dta.
    """
    run_state: RunState = request.registry["run_state"]

    r = Response(content_type="application/json")
    r.text = run_state.make_exportable_copy().to_json()
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

        r = Response(content_type="image/png")
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

