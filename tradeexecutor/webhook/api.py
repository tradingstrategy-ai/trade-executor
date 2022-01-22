import pkg_resources

from pyramid.request import Request
from pyramid.response import Response
from pyramid.view import view_config


@view_config(route_name='home', renderer='json', permission='view')
def home(request: Request):
    url = request.application_url
    version = pkg_resources.get_distribution('tradeexecutor').version
    # https://angrybirds.fandom.com/wiki/The_Flock
    return Response(f'Chuck Trade Executor server, version {version}, our URL is {url}')


@view_config(route_name='ping', renderer='json', permission='view')
def ping(request: Request):
    """Unauthenticated endpoint to check the server is up."""
    return {"ping": "pong"}


@view_config(route_name='notify', renderer='json', permission='view')
def notify(request: Request):
    """Notify the strategy executor about the availability of new data."""
    return {"status": "OK"}