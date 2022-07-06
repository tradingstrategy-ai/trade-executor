"""Pyramid web framework event handlers."""

from pyramid.events import NewResponse, subscriber


@subscriber(NewResponse)
def add_cors_headers(event):
    """Add CORS headers.

    See https://stackoverflow.com/a/47167858/315168.
    """
    event.response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST,GET,DELETE,PUT,OPTIONS',
        'Access-Control-Allow-Headers': 'Origin, Content-Type, Accept, Authorization',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '1728000',
    })