"""Public API for recording decision-relevant live strategy inputs.

Strategies opt in through ``Parameters.record_strategy_inputs``. The live
pandas runner then exposes a :class:`DecisionRecorder` as
``StrategyInput.recorder``. A strategy explicitly starts, records, and closes
each decision; the recorder never changes trade selection or executor state.
"""

from tradeexecutor.strategy.recorder.recorder import DecisionRecorder

__all__ = ["DecisionRecorder"]
