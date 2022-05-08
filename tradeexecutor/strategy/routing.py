"""Trade routing instructions.

Each trading universe and strategy can have different trade routing set,
based on the exchanges the universe covers.

Here we define the abstract overview of routing.
"""
import abc


class RoutingModel(abc.ABC):
    """Trade roouting model base class.

    Nothing done here - check the subclasses.
    """