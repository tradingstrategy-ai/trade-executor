"""Pyramid web server app configuration.

See https://docs.pylonsproject.org/projects/pyramid-cookbook/en/latest/auth/basic.html for
HTTP Basic Auth setup.
"""

import os
import logging
from queue import Queue

from pyramid.authentication import BasicAuthAuthenticationPolicy
from pyramid.authorization import ACLAuthorizationPolicy, Allow, Authenticated
from pyramid.config import Configurator
from pyramid.httpexceptions import HTTPUnauthorized, HTTPForbidden
from pyramid.router import Router
from pyramid.security import forget, ALL_PERMISSIONS
from pyramid.view import forbidden_view_config

from . import api
from .error import exception_view


logger = logging.getLogger(__name__)


@forbidden_view_config()
def forbidden_view(request):
    if request.authenticated_userid is None:
        response = HTTPUnauthorized()
        response.headers.update(forget(request))

    # user is logged in but doesn't have permissions, reject wholesale
    else:
        response = HTTPForbidden()
    return response



class Root:
    # dead simple, give everyone who is logged in any permission
    # (see the home_view for an example permission)
    __acl__ = (
        (Allow, Authenticated, ALL_PERMISSIONS),
    )


def create_pyramid_app(username, password, command_queue: Queue, production=False) -> Router:
    """Create WSGI app for Trading Strategy backend."""

    settings = {
        'production': production,
        'command_queue': command_queue,
    }

    def check_credentials(username_, password_, request):
        if username == username_ and password == password_:
            # an empty list is enough to indicate logged-in... watch how this
            # affects the principals returned in the home view if you want to
            # expand ACLs later
            return []

    with Configurator(settings=settings) as config:

        authn_policy = BasicAuthAuthenticationPolicy(check_credentials)
        config.set_authentication_policy(authn_policy)
        config.set_authorization_policy(ACLAuthorizationPolicy())
        config.set_root_factory(lambda request: Root())

        config.add_route('home', '/')
        config.add_view(api.home, route_name='home')

        config.add_route('ping', '/ping')
        config.add_view(api.ping, route_name='ping', renderer="json")

        config.add_route('notify', '/notify')
        config.add_view(api.notify, route_name='notify', renderer="json")

        config.add_exception_view(exception_view)

        if production:
            # Because datadog import modifies the global process and messes up things,
            # do not touch it if not absoluteneeded
            # https://docs.datadoghq.com/tracing/setup_overview/setup/python/?tab=containers
            if os.environ.get("DD_ENV") == "prod":
                logger.info("Enabling Datadog tracing")
                import ddtrace
                from ddtrace.contrib.pyramid import trace_pyramid
                ddtrace.patch(psycopg=True)
                trace_pyramid(config)

        app = config.make_wsgi_app()
        return app
