"""High-grade logging facilities.

We have a custom level `logging.TRADE` that we use to log trade execution to Discord and such.
"""

import logging
import sys
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Optional, List

from tradeexecutor.cli.version_info import VersionInfo
from tradeexecutor.utils.ring_buffer_logging_handler import RingBufferHandler

try:
    import coloredlogs
    import logstash
    from discord_logging.handler import DiscordHandler
except ImportError:
    # Not available for Pyodide build, see
    # test_optional_dependencies.py
    pass


#: Stored here as a global so that we can later call RingBufferHandler.export()
_ring_buffer_handler: Optional[RingBufferHandler] = None


def setup_logging(
    log_level: None | str | int=logging.INFO,
    in_memory_buffer=False,
    enable_trade_high=False,
) -> Logger:
    """Setup root logger and quiet some levels.

    :param log_level:
        Log level read from command line or environment var.

    :param in_memory_buffer:
        Setup in-memory log buffer used to fetch log messages to the frontend.
    """
    setup_custom_log_levels(enable_trade_high=enable_trade_high)

    if log_level == "disabled":
        # Special unit test marker, don't mess with loggers
        return logging.getLogger()
    elif log_level is None:
        log_level = logging.INFO

    if isinstance(log_level, str):
        log_level = log_level.upper()

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

    # Because we are running a single threaded worker,
    # the position trigger check task can be blocked by trade decision task, or vice versa.
    # Because the position trigger check task is very quick, this should not be an issue in practice.
    # However apscheduler spews logs with WARN on this and we do not want it as
    # warning, as it is a planned scenario.
    # enzyme-polygon-eth-usdc  | 2023-06-23 13:00:05 apscheduler.executors.default                      WARNING  Run time of job "ExecutionLoop.run_live.<locals>.live_positions (trigger: interval[1:00:00], next run at: 2023-06-23 14:00:00 UTC)" was missed by 0:00:05.324013
    logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)

    # GraphQL library
    # Writes very noisy and long Introspection query to logs
    # everytime graphql endpoint is read
    logging.getLogger("graphql").setLevel(logging.WARNING)
    logging.getLogger("gql").setLevel(logging.WARNING)

    # By default, disable performance monitor logging
    logging.getLogger("tradeexecutor.utils.timer").setLevel(logging.WARNING)

    # 2024-09-26 12:11:25 traitlets WARNING  Alternative text is missing on 3 image(s).
    logging.getLogger("traitlets").setLevel(logging.ERROR)

    if in_memory_buffer:
        setup_in_memory_logging(logger)

    return logger


def setup_file_logging(
    log_filename: str | Path,
    log_level: str | int = logging.INFO,
    http_logging=False,
):
    """Create log file output.

    - All log messages

    - HTTP traffic only

    """
    # https://stackoverflow.com/a/11111212/315168

    fmt = "%(asctime)s %(name)-50s %(levelname)-8s %(message)s"
    formatter = logging.Formatter(fmt)

    if isinstance(log_level, str):
        log_level = log_level.upper()

    if log_level == "NONE":
        # Allow disable
        return

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    logging.getLogger().addHandler(file_handler)

    if http_logging:
        # Delayed import, as we do not want to bring in web server machinery in backtesting

        from tradeexecutor.webhook.http_log import configure_http_request_logging
        # Create a logger for HTTP requests only
        configure_http_request_logging(Path(log_filename))


def setup_in_memory_logging(logger):
    global _ring_buffer_handler
    _ring_buffer_handler = RingBufferHandler(logging.INFO)
    logger.addHandler(_ring_buffer_handler)


def get_ring_buffer_handler() -> RingBufferHandler:
    return _ring_buffer_handler


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

    # Don't log section duration monitors
    logging.getLogger("tradeexecutor.utils.timer").setLevel(logging.WARNING)

    # Mute mainloop (a lot of logs about nothing happening in backtesting)
    # logging.getLogger("tradeexecutor.strategy.runner").setLevel(logging.WARNING)

    return logging.getLogger("test")


def setup_notebook_logging(log_level: str | int=logging.WARNING, show_process=False) -> logging.Logger:
    """Setup logger in notebook / backtesting environments.

    This will enable logging for all loggeres and the output is too verbose
    for backtesting.
    """

    logger = logging.getLogger()

    # Set log format to dislay the logger name to hunt down verbose logging modules
    if show_process:
        # Debugging the Loky backing - what a mess
        format = "<proc:%(process)d> %(asctime)s %(name)-50s %(levelname)-8s %(message)s"
    else:
        format = "%(asctime)s %(name)-50s %(levelname)-8s %(message)s"

    # TODO: coloredlogs disabled for notebook -
    # see https://stackoverflow.com/a/68930736/315168
    # how to add native HTML log output
    # Use colored logging output for console
    # coloredlogs.install(level=log_level, fmt=fmt, logger=logger)

    if isinstance(log_level, str):
        log_level = log_level.upper()

    logger = logging.getLogger()

    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        format=format,
    )
    #logger.setLevel(log_level)

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

    # Performance metrics, irrelevant for backtesting
    logging.getLogger("tradeexecutor.utils.timer").setLevel(logging.WARNING)

    return logger


def setup_strategy_logging(default_log_level: str | int=logging.WARNING) -> logging.Logger:
    """Setup logging for backtesting.

    This will enable logging :py:class:`~tradeexecutor.strategy.pandas_trading.PositionManager` only.

    See also :py:meth:`tradeexecutor.strategy.pandas_trading.position_manager.PositionManager.log`
    for examples.
    """

    # TODO: coloredlogs disabled for notebook -
    # see https://stackoverflow.com/a/68930736/315168
    # how to add native HTML log output
    # Use colored logging output for console
    # coloredlogs.install(level=log_level, fmt=fmt, logger=logger)

    if isinstance(default_log_level, str):
        default_log_level = default_log_level.upper()

    # See https://stackoverflow.com/a/56532290/315168
    logger = logging.getLogger()
    logging.basicConfig(stream=sys.stdout, level=default_log_level)
    setup_custom_log_levels()
    logging.getLogger("tradeexecutor.strategy.pandas_trader.position_manager").setLevel(logging.INFO)
    return logger


def setup_custom_log_levels(
    enable_trade_high=False,
):
    """Create a new logging level TRADE that is between INFO and WARNING.

    This level is used to log trade execution to Discord etc.
    trader followed stream.

    Add

    - `logging.TRADE`: Log level for verbose trade output for Discord diagnostics

    - `logging.TRADE_HIGH`: Log level made trade decisions - this only logs
       successful trade decisions, not any errors. It is designed for the Teleegram bot.

    :param enable_trade_high:
        Enable special TRADE_HIGH logging level in live trade execution.
    """

    if hasattr(logging, "TRADE"):
        # Already setup, don't try twice
        return

    # https://www.programcreek.com/python/?code=dwavesystems%2Fdwave-hybrid%2Fdwave-hybrid-master%2Fhybrid%2F__init__.py

    # Log level for verbose trade output for Discord diagnostics
    logging.TRADE = logging.INFO + 1  # Info is 20, TRADE is 21, Warning is 30

    if enable_trade_high:
        # A level that shows trade output but no errors in live exec
        logging.TRADE_HIGH = logging.FATAL + 1
    else:
        # Otherwise the same as trade
        logging.TRADE_HIGH = logging.TRADE + 1

    # Log level
    logging.addLevelName(logging.TRADE, "TRADE")
    logging.addLevelName(logging.TRADE_HIGH, "TRADE_HIGH")

    def _trade(logger, message, *args, **kwargs):
        if logger.isEnabledFor(logging.TRADE):
            logger._log(logging.TRADE, message, args, **kwargs)

    def _trade_high(logger, message, *args, **kwargs):
        if logger.isEnabledFor(logging.TRADE_HIGH):
            logger._log(logging.TRADE_HIGH, message, args, **kwargs)

    logging.Logger.trade = _trade
    logging.Logger.trade_high = _trade_high


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


def setup_telegram_logging(
    telegram_api_key: str,
    telegram_chat_id: str,
) -> "telegram_bot_logger.TelegramMessageHandler":
    """Setup Telegram logger.

    Set up a Python logging handler based on `telegram_bot_logger <https://github.com/arynyklas/telegram_bot_logger>`__
    to send trade output to a Telegram group chat.

    .. note::

        This handler spawns a background thread. You need to call `handler.close` or your application won't exit.

    **Manual testing instructions**.

    Invite the bot to a group chat. Then send a message `/start @botname` to the bot in the group chat to activate it.

    Then get chat id with:

    .. code-block:: shell

         curl https://api.telegram.org/bot$TELEGRAM_API_KEY/getUpdates | jq

    Test with:

    .. code-block:: shell

        pytest --log-cli-level=info -k test_telegram_logging

    - `More Telegram bot set up details <https://stackoverflow.com/questions/64990028/how-to-send-a-message-to-telegram-from-zapier/64990029#64990029>`__

    """

    assert telegram_api_key
    assert telegram_chat_id

    import telegram_bot_logger
    from telegram_bot_logger.formatters import TelegramHTMLTextFormatter

    # Fine tune our Telegram chat output
    formatter = TelegramHTMLTextFormatter()
    formatter._EMOTICONS[logging.TRADE] = "💰"  # Patch in the custom log level
    formatter._EMOTICONS[logging.TRADE_HIGH] = "💰"  # Patch in the custom log level
    formatter._TAG_FORMAT = "" # Disable tags in the output
    formatter._HEADER_FORMAT = "{emoticon} {message} {description}"  # Disable line no + module in the output

    telegram_handler = telegram_bot_logger.TelegramMessageHandler(
        bot_token=telegram_api_key,  # Required; bot's token from @BotFather
        chat_ids=[
            int(telegram_chat_id)  # Make sure group chat ids are integer
        ],
        format_type="text",
        formatter=formatter,
        level=logging.TRADE_HIGH,
    )

    logging.getLogger().addHandler(telegram_handler)
    logging.getLogger().info("Telegram logging activated, chat id is %s", telegram_chat_id)
    return telegram_handler


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


def setup_sentry_logging(*, application_name: str, sentry_dsn: str):
    """Setup Sentry logging.

    :param sentry_dsn:
        Sentry DSN
    """
    import sentry_sdk
    from sentry_sdk.types import Breadcrumb, BreadcrumbHint

    docker_version_info = VersionInfo.read_docker_version()

    def before_breadcrumb(crumb: Breadcrumb, hint: BreadcrumbHint) -> Breadcrumb | None:
        # Ignore httplib logs, since they are Discord requests
        if crumb["category"] == "httplib":
            return None

        # Sentry doesn't support our custom log level, so we need to map it to INFO
        custom_level_mapping = {
            "trade": "info",
            "trade_high": "info",
        }
        current_level = crumb.get("level")
        crumb["level"] = custom_level_mapping.get(current_level, current_level)

        return crumb

    # Add our Docker build tag in th Sentry event data,
    # so we know which trade-executor version caused error in Sentry
    def before_send(event, hint):
        event["extra"] = event.get("extra", {})
        event["extra"]["docker_image_version"] = docker_version_info.tag
        event["extra"]["docker_image_commit_message"] = docker_version_info.commit_message
        event["extra"]["docker_image_commit_hash"] = docker_version_info.commit_hash
        return event

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=application_name,
        traces_sample_rate=0.1,
        before_breadcrumb=before_breadcrumb,
        before_send=before_send,
    )
