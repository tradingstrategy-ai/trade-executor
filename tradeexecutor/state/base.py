import abc


class BaseStore(abc.ABC):

    def load_state(self):
        pass

    def save_state(self):
        pass