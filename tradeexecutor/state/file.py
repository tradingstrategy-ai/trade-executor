"""Serialize the strategy state to a local file."""
from tradeexecutor.state.base import BaseStore


class FileStore(BaseStore):

    def __init__(self, path):
        self.path = path

    def save(self):
        pass

    def preflight_check(self):
        pass