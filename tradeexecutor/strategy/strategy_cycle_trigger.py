"""Strategy cycle trigger.

How do we decide when to run the next strategy cycle.
"""
import enum


class StrategyCycleTrigger(enum.Enum):
    """How do we decide when to run the next live strategy cycle."""

    #: Offset time.
    #:
    #: Wait a fixed time after the decision timestamp has been reached
    #: before attempting to download new data and execute decisions based on it.
    cycle_offset = "cycle_offset"


    #: Trading pair data availability.
    #:
    #: Poll trading pair data availability endpoint and
    #: immediately attempt to execute live trading cycle when new data is available.
    trading_pair_data_availability = "trading_pair_data_availability"