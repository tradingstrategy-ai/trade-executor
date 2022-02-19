import logging

import coloredlogs
from _pytest.fixtures import SubRequest


def setup_pytest_logging(request: SubRequest) -> logging.Logger:
    """Setup logger in pytest environment.

    Quiets out unnecessary logging subsystems.
    """

    # Disable logging of JSON-RPC requests and reploes
    logging.getLogger("web3.RequestManager").setLevel(logging.WARNING)
    logging.getLogger("web3.providers.HTTPProvider").setLevel(logging.WARNING)

    # Disable all internal debug logging of requests and urllib3
    # E.g. HTTP traffic
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # ipdb internals
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return logging.getLogger("test")