"""API function entrypoints."""

import os
import logging
from importlib.metadata import version

from pyramid.request import Request
from pyramid.response import Response, FileResponse
from pyramid.view import view_config

from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.execution_state import ExecutionState
from tradeexecutor.webhook.error import exception_response


logger = logging.getLogger(__name__)


@view_config(route_name='home', permission='view')
def web_home(request: Request):
    """/ endpoint.

    The homepage displays plain text version banner.
    """
    url = request.application_url
    version_ = version('trade-executor')
    # https://angrybirds.fandom.com/wiki/The_Flock
    return Response(f'Chuck the Trade Executor server, version {version_}, our URL is {url}\nFor more information see https://tradingstrategy.ai\nRemember to play Angry Birds.', content_type="text/plain")


@view_config(route_name='web_ping', renderer='json', permission='view')
def web_ping(request: Request):
    """/ping endpoint

    Unauthenticated endpoint to check the serverPlain is up.
    """
    return {"ping": "pong"}


@view_config(route_name='web_metadata', permission='view')
def web_metadata(request: Request):
    """/metadata endpoint

    Executor metadata.
    """
    metadata: Metadata = request.registry["metadata"]
    execution_state: ExecutionState = request.registry["execution_state"]

    # Retrofitted with the running flag,
    # not really a nice API design.
    # Do not mutate a global state in place/
    metadata = Metadata(**metadata.to_dict())
    metadata.executor_running = execution_state.executor_running

    r = Response(content_type="application/json")
    r.body = metadata.to_json().encode("utf-8")
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
    execution_state: ExecutionState = request.registry["execution_state"]
    r = Response(content_type="application/json")
    r.body = execution_state.to_json().encode("utf-8")
    return r
