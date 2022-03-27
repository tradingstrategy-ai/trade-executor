"""Webhook web server."""
import logging
from queue import Queue

from waitress.server import create_server, MultiSocketServer
from webtest.http import StopableWSGIServer

from .app import create_pyramid_app


logger =  logging.getLogger(__name__)


class Server(StopableWSGIServer):

    def close_for_use(self):
        pass


def create_webhook_server(host: str, port: int, username: str, password: str, queue: Queue) -> StopableWSGIServer:
    """Starts the webhook web  server in a separate thread.

    :param queue: The command queue for commands posted in the webhook that offers async execution.
    """

    assert username, "Username must be given"
    assert password, "Password must be given"

    app = create_pyramid_app(username, password, queue, production=False)
    server = StopableWSGIServer.create(app, host=host, port=port, clear_untrusted_proxy_headers=True)
    logger.info("Webhook server will spawn at %s:%d", host, port)
    # Wait until the server has started
    server.wait()
    return server

    #port = server.adj.port # 8521
    #server.wait()
    #url = f"http://{host}:{port}"
    #logger.info("Webhook server spawned at %s", url)
    #yield url
    #server.shutdown()

