import pkg_resources

from pyramid.request import Request
from pyramid.response import Response
from pyramid.view import view_config


@view_config(route_name='home', renderer='json', permission='view')
def web_home(request: Request):
    url = request.application_url
    version = pkg_resources.get_distribution('tradeexecutor').version
    # https://angrybirds.fandom.com/wiki/The_Flock
    return Response(f'Chuck the Trade Executor server, version {version}, our URL is {url}')


@view_config(route_name='web_ping', renderer='json', permission='view')
def web_ping(request: Request):
    """Unauthenticated endpoint to check the server is up."""
    return {"ping": "pong"}


@view_config(route_name='web_notify', renderer='json', permission='view')
def web_notify(request: Request):
    """Notify the strategy executor about the availability of new data."""
    # TODO
    return {"status": "OK"}


@view_config(route_name='web_state', renderer='json', permission='view')
def web_state(request: Request):
    """Notify the strategy executor about the availability of new data."""
    # TODO
    return {"status": "OK"}
