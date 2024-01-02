"""Web server logging.

"""
import datetime
import logging
from pathlib import Path
from urllib.request import Request

from pyramid.registry import Registry

logger = logging.getLogger(__name__)

_req_id_country = 0


def log_tween_factory(handler, registry: Registry):

    def log_tween(request: Request):

        global _req_id_country

        _req_id_country += 1
        req_id = _req_id_country

        country = request.headers.get("CF-IPCountry")
        ip_addr = request.headers.get("CF-Connecting-IP")

        logger.info("HTTP request #%d %s (%s): %s", req_id, ip_addr, country, request.full_url)

        start = datetime.datetime.utcnow()
        try:
            response = handler(request)
            end = datetime.datetime.utcnow()
            duration = end - start
            logger.info("HTTP response #%d duration:% %s", req_id, duration, request.full_url)
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
