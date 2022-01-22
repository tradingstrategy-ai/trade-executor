import pkg_resources

from pyramid.request import Request
from pyramid.response import Response


def home(request: Request):
    url = request.application_url
    version = pkg_resources.get_distribution('tradeexecutor').version
    # https://angrybirds.fandom.com/wiki/The_Flock
    return Response(f'Chuck Trade Executor server, version {version}, our URL is {url}')


def ping(request: Request):
    """Unauthenticated endpoint to check the server is up."""
    return {"ping": "pong"}


def notify(request: Request):
    """Notify the strategy executor about the availability of new data."""
    return {"status": "OK"}