from tradeexecutor.state.base import BaseStore


class InMemoryStore(BaseStore):
    """The current state is only stored in the process memory."""

    def __init__(self):
        pass

    def save(self):
        pass

    def preflight_check(self):
        pass