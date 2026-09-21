"""Expose the strategy-facing API for live decision-input recording.

Strategies opt in by decorating ``decide_trades()`` with ``@record_decision``. The live
pandas runner then exposes a :class:`DecisionRecorder` as
``StrategyInput.recorder``. A strategy applies :func:`record_decision` to its
``decide_trades()`` callback and records only decision-specific observations;
framework bootstrap constructs and closes the underlying recorder. Capture and
storage submodules are internal implementation details, so strategies should
import only from this package.
"""

from tradeexecutor.strategy.recorder.recorder import DecisionRecorder, record_decision

__all__ = ["DecisionRecorder", "record_decision"]
