"""Calculate summary statistics on background.

"""
from tradeexecutor.state.state import State


def run_background_statistics_task(
    state: State,
    func,
    *args,
    **kwargs,
):
    """Run a background statistics task.

    Only one task can run at a time, on a background.
    If there are several of piled up tasks,
    only the first one will
    """
    raise NotImplementedError()


