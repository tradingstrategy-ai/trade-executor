"""High-grade logging facilities.

We have a custom level `logging.TRADE` that we use to log trade execution to Discord and such.
"""

import logging
from logging import Logger
from typing import Optional, List

try:
    import coloredlogs
    import logstash
    from discord_logging.handler import DiscordHandler
except ImportError:
    # Not available for Pyodide build, see
    # test_optional_dependencies.py
    pass

def setup_logging(log_level=logging.INFO) -> Logger:
    """Setup root logger and quiet some levels."""

    setup_custom_log_levels()

    logger = logging.getLogger()

    # Set log format to dislay the logger name to hunt down verbose logging modules
    fmt = "%(asctime)s %(name)-50s %(levelname)-8s %(message)s"

    # Use colored logging output for console
    coloredlogs.install(level=log_level, fmt=fmt, logger=logger)

    # Disable logging of JSON-RPC requests and reploes
    logging.getLogger("web3.RequestManager").setLevel(logging.WARNING)
    logging.getLogger("web3.providers.HTTPProvider").setLevel(logging.WARNING)
    # logging.getLogger("web3.RequestManager").propagate = False

    # Disable all internal debug logging of requests and urllib3
    # E.g. HTTP traffic
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # IPython notebook internal
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Maplotlib puke
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Disable warnings on startup
    logging.getLogger("pyramid_openapi3").setLevel(logging.ERROR)

    # Datadog tracer agent
    # https://ddtrace.readthedocs.io/en/stable/basic_usage.html
    logging.getLogger("ddtrace").setLevel(logging.INFO)

    return logger


def setup_pytest_logging(request=None, mute_requests=True) -> logging.Logger:
    """Setup logger in pytest environment.

    Quiets out unnecessary logging subsystems.

    :param request:
        pytest.fixtures.SubRequest instance

    :return:
        Test logger - though please use module specific logger
    """

    setup_custom_log_levels()

    # Disable logging of JSON-RPC requests and reploes
    logging.getLogger("web3.RequestManager").setLevel(logging.WARNING)
    logging.getLogger("web3.providers.HTTPProvider").setLevel(logging.WARNING)

    # Disable all internal debug logging of requests and urllib3
    # E.g. HTTP traffic
    if mute_requests:
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ipdb internals
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # maplotlib burps a lot on startup
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logging.getLogger("test")


def setup_notebook_logging(log_level=logging.WARNING) -> logging.Logger:
    """Setup logger in notebook / backtesting environments."""

    logger = logging.getLogger()

    # Set log format to dislay the logger name to hunt down verbose logging modules
    fmt = "%(asctime)s %(name)-50s %(levelname)-8s %(message)s"

    # TODO: coloredlogs disabled for notebook -
    # see https://stackoverflow.com/a/68930736/315168
    # how to add native HTML log output
    # Use colored logging output for console
    # coloredlogs.install(level=log_level, fmt=fmt, logger=logger)

    setup_custom_log_levels()

    # Disable logging of JSON-RPC requests and reploes
    logging.getLogger("web3.RequestManager").setLevel(logging.WARNING)
    logging.getLogger("web3.providers.HTTPProvider").setLevel(logging.WARNING)

    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ipdb internals
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # maplotlib burps a lot on startup
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger


def setup_custom_log_levels():
    """Create a new logging level TRADE that is between INFO and WARNING.

    This level is used to log trade execution to Discord etc.
    trader followed stream.
    """

    if hasattr(logging, "TRADE"):
        # Already setup, don't try twice
        return

    # https://www.programcreek.com/python/?code=dwavesystems%2Fdwave-hybrid%2Fdwave-hybrid-master%2Fhybrid%2F__init__.py
    logging.TRADE = logging.INFO + 1  # Info is 20, Warning is 30
    logging.addLevelName(logging.TRADE, "TRADE")

    def _trade(logger, message, *args, **kwargs):
        if logger.isEnabledFor(logging.TRADE):
            logger._log(logging.TRADE, message, args, **kwargs)

    logging.Logger.trade = _trade


def setup_discord_logging(name: str, webhook_url: str, avatar_url: Optional[str]=None):
    """Setup Discord logger.

    Any log with level `logging.TRADE` is echoed to Discord channel given in the URL.

    https://pypi.org/project/discord-logger/

    :param url:
    :return:
    """
    discord_format = logging.Formatter("%(message)s")
    discord_handler = DiscordHandler(name, webhook_url, avatar_url=avatar_url, message_break_char="…")
    discord_handler.setFormatter(discord_format)
    discord_handler.setLevel(logging.TRADE)
    logging.getLogger().addHandler(discord_handler)


def setup_logstash_logging(
        logstash_server: str,
        application_name: str,
        extra_tags: Optional[List[str]] = None,
        quiet=True,
        level=logging.INFO,
        port=5959,
):
    """Setup Logstash logger.

    Connects via UDP.

    :param logstash_server:
        Host name / IP address of a logstash server

    :param port:
        Logstash receiving UDP port

    :param application_name:
        Added as application tag.

        You can look up this application by filtering for `application`
        in Logstash.

    :param extra_tags:
        List of extra tags added on on Logstash messages.

    :param quiet:
        Do we note if we have connected to the server

    :return:
        Logstash logger
    """

    logger = logging.getLogger()

    extra_tags = extra_tags or []

    tags = ["python"] + extra_tags
    if not quiet:
        logger.info("Logging to Logstash server %s, application name is %s, tags are %s",
                    logstash_server,
                    application_name, tags)
    # Include our application name in the logs
    assert application_name, "Cannot use Logstash without application_name set"
    extra_fields = {"application": application_name}
    handler = logstash.UDPLogstashHandler(logstash_server, port, version=1, tags=tags, extra_fields=extra_fields)
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger
