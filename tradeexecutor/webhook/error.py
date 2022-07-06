"""Web server error management."""
import json
import logging

from pyramid.httpexceptions import status_map
from pyramid.request import Request
from pyramid.response import Response


logger = logging.getLogger(__name__)


def convert_to_api_error_response(e: Exception, code=401):

    logger.error("An error in API endpoint: %s", e)
    logger.exception(e)

    data = json.dumps({"error": str(e), "class": str(e.__class__)})
    return Response(data, status=code, content_type="application/json", charset="utf-8")


def error_tween_factory(handler, registry):
    # See https://github.com/Pylons/pyramid_exclog/blob/master/pyramid_exclog/__init__.py
    def error_tween_factory(request):
        try:
            response = handler(request)
            exc_info = getattr(request, 'exc_info', None)
            if exc_info is not None:
                # Never commit on exception
                logger.info("Aborting the transaction")
                # rollback_all_sessions(request)
                return convert_to_api_error_response(exc_info)
            return response
        except Exception as e:
            # rollback_all_sessions(request)
            logger.info("Error handler borked out %s", e)
            logger.exception(e)
            return convert_to_api_error_response(e)
    return error_tween_factory


def exception_view(exc: Exception, request: Request):
    # rollback_all_sessions(request)
    logger.info("Error handler borked out %s", exc)
    logger.exception(exc)
    return convert_to_api_error_response(exc)


def exception_response(status_code, **kw):
    """Creates an HTTP exception based on a status code.

    The values passed as ``kw`` are provided to the exception's constructor.

    Example:

    .. code-block:: python

        return exception_response(404, detail="Status file not yet created")

    Example:

    .. code-block:: python

        raise exception_response(404) # raises an HTTPNotFound exception.

    """
    logger.warning("Web server returned an error: %d %s", status_code, kw)
    exc = status_map[status_code](**kw)
    return exc
