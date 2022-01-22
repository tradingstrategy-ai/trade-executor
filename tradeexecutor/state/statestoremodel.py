import enum


class StateStoreModel(enum.Enum):
    """How the algorithm execution state is stored."""

    file = "file"

    on_chain = "on_chain"