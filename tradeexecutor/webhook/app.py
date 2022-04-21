"""Pyramid web server app configuration.

See https://docs.pylonsproject.org/projects/pyramid-cookbook/en/latest/auth/basic.html for
HTTP Basic Auth setup.
"""

import os
import logging
import warnings
from queue import Queue

from pyramid.authentication import BasicAuthAuthenticationPolicy
from pyramid.authorization import ACLAuthorizationPolicy, Allow, Authenticated
from pyramid.config import Configurator
from pyramid.httpexceptions import HTTPUnauthorized, HTTPForbidden
from pyramid.router import Router
from pyramid.security import forget
from pyramid.authorization import ALL_PERMISSIONS
from pyramid.view import forbidden_view_config

from . import api
from .error import exception_view
from ..state.metadata import Metadata
from ..state.store import JSONFileStore

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


def init_web_api(config: Configurator):
    """Setup OpenAPI endpoints for the website backend."""

    # Github submodule contents
    cur_path = os.path.dirname(__file__)
    web_spec = os.path.join(cur_path, "..", "..", "spec", "trade-executor-api.yaml")
    web_spec = os.path.abspath(web_spec)
    assert os.path.exists(web_spec), f"Web spec missing {web_spec}, did you do recursive git checkout?"

    config.include("pyramid_openapi3")

    # config.pyramid_openapi3_spec_directory(web_spec, route='/onchain')
    config.pyramid_openapi3_spec_directory(web_spec, route='/api')

    config.registry.settings["pyramid_openapi3.enable_endpoint_validation"] = False
    config.registry.settings["pyramid_openapi3.enable_request_validation"] = False
    config.registry.settings["pyramid_openapi3.enable_response_validation"] = False

    config.pyramid_openapi3_register_routes()

    config.scan(package='tradeexecutor.webhook.api')
    config.scan(package='tradeexecutor.webhook.events')


def create_pyramid_app(username, password, command_queue: Queue, store: JSONFileStore, metadata: Metadata, production=False) -> Router:
    """Create WSGI app for Trading Strategy backend."""

    settings = {
        'production': production,
        'command_queue': command_queue,
    }

    # Run the server without HTTP Basic Auth
    http_basic_auth = (username) and (password)

    def check_credentials(username_, password_, request):
        if username == username_ and password == password_:
            # an empty list is enough to indicate logged-in... watch how this
            # affects the principals returned in the home view if you want to
            # expand ACLs later
            return []

    with Configurator(settings=settings) as config:

        if http_basic_auth:
            authn_policy = BasicAuthAuthenticationPolicy(check_credentials)

            # https://stackoverflow.com/a/14463362/315168
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # TODO: Upgrade to Pyramid 2.0
                config.set_authentication_policy(authn_policy)
                config.set_authorization_policy(ACLAuthorizationPolicy())

            config.set_root_factory(lambda request: Root())
        else:
            logger.info("Authentication policy disabled")

        init_web_api(config)

        # Expose the state store to the webhook
        config.registry["store"] = store
        config.registry["metadata"] = metadata

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
