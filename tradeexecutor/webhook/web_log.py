"""Web server logging.

- Output HTTP request and response metadata to a separate log file

"""
import datetime
import logging
import threading
from pathlib import Path

from pyramid.registry import Registry
from pyramid.request import Request

logger = logging.getLogger(__name__)

# Never propagate web logs as root logging output is assuemed to be public.
# IP addresses and such are logged to their own file.
# https://stackoverflow.com/a/67364351/315168
logger.propagate = False

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

        country = request.headers.get("CF-IPCountry")
        ip_addr = request.headers.get("CF-Connecting-IP")

        logger.info("HTTP request #%d %s (%s): %s", req_id, ip_addr, country, request.url)

        start = datetime.datetime.utcnow()
        try:
            response = handler(request)
            end = datetime.datetime.utcnow()
            duration = end - start
            logger.info("HTTP response #%d duration:% %s", req_id, duration, request.url)
            return response
        except Exception as e:
            logger.info("HTTP response failed: %s", e)
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

    log_path = main_log_path.rename(main_log_path.with_suffix(".http.log"))

    fmt = "%(asctime)s %(message)s"
    formatter = logging.Formatter(fmt)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.handlers.clear()
    logger.addHandler(file_handler)

    return logger
