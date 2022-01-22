import enum


class StateModel(enum.Enum):
    """How the algorithm execution state is stored."""

    file = "file"

    on_chain = "on_chain"