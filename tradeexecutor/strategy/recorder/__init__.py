"""Expose the strategy-facing API for live decision-input recording.

Strategies opt in through ``Parameters.record_strategy_inputs``. The live
pandas runner then exposes a :class:`DecisionRecorder` as
``StrategyInput.recorder``. A strategy explicitly starts, records, and closes
each decision from ``decide_trades()``; framework bootstrap imports the same
class to construct and close it. Capture and storage submodules are internal
implementation details, so strategies should import only from this package.
"""

from tradeexecutor.strategy.recorder.recorder import DecisionRecorder

__all__ = ["DecisionRecorder"]
