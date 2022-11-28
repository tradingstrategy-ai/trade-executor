from tradeexecutor.strategy.execution_state import ExecutionState


def test_serialise_exception():
    """Make sure we can JSON serialise exceptions."""
    try:
        raise RuntimeError("Boom")
    except Exception as e:
        exception_data = ExecutionState.serialise_exception()

    assert exception_data["exception_message"] == "Boom"
    assert "tb_frame" in exception_data
