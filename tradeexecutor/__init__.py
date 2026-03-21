import sys

from tradeexecutor.utils.pandas_backend import configure_pandas_arrow_backend

assert sys.version_info >= (3, 3), "Python 3.9 version minimun to run trade-executor"

configure_pandas_arrow_backend()
