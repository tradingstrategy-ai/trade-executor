
def test_optional_dependencies():
    """See that we can import the library without optional dependencies.

    See .github/workflows/client-side.yml
    """

    # Make sure we can import core modules under Pyodide
    from tradeexecutor.state import state
    from tradeexecutor.statistics import core
    from tradeexecutor.visual import single_pair

