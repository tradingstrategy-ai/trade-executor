"""Web server logging.

- Output HTTP request and response metadata to a separate log file

"""
import datetime
import logging
import threading
from pathlib import Path

from pyramid.registry import Registry
from pyramid.request import Request

# Avoid logging.getLogger() here as we do not want this logger as the part of std logging system
http_logger = logging.Logger(name="HTTP traffic")

# Never propagate web logs as root logging output is assuemed to be public.
# IP addresses and such are logged to their own file.
# https://stackoverflow.com/a/67364351/315168
http_logger.propagate = False

#: Unique id for every request - response pair
_req_id_country = 0
_req_lock = threading.Lock()


def log_tween_factory(handler, registry: Registry):

    def log_tween(request: Request):

        global _req_id_country

        with _req_lock:
            # Should not timeout ever
            _req_id_country += 1  # Not atomic https://stackoverflow.com/questions/1717393/is-the-operator-thread-safe-in-python
            req_id = _req_id_country

        country = request.headers.get("CF-IPCountry") or "<no country>"
        ip_addr = request.headers.get("CF-Connecting-IP") or "<no CF IP>"
        user_agent = request.user_agent

        http_logger.info("HTTP request #%d %s (%s): %s by %s", req_id, ip_addr, country, request.url, user_agent)

        start = datetime.datetime.utcnow()
        try:
            response = handler(request)
            end = datetime.datetime.utcnow()
            duration = end - start
            http_logger.info("HTTP response #%d duration:%s %s", req_id, duration, request.url)
            return response
        except Exception as e:
            http_logger.error("HTTP response #%d failed: %s", req_id, e)
            raise

    return log_tween


def configure_http_request_logging(main_log_path: Path) -> logging.Logger:
    """Configure HTTP requests to be logged to a separate file.

    Must be called after other loggers have been configured.

    :param main_log_path:
        Trade execution log file.

        We will prepare a log file with HTTP specific entries.
    """
    assert isinstance(main_log_path, Path)

    log_path = main_log_path.with_suffix(".http.log")

    fmt = "%(asctime)s %(message)s"
    formatter = logging.Formatter(fmt)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    http_logger.handlers.clear()
    http_logger.addHandler(file_handler)

    http_logger.info("Starting HTTP traffic log at %s", log_path)

    return http_logger
