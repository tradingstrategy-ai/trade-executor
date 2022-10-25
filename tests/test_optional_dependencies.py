import pytest


def test_optional_dependencies():
    """See that we can import the library without optional dependencies."""

    from tradeexecutor.state import state
    from tradeexecutor.statistics import core
    from tradeexecutor.visual import single_pair

