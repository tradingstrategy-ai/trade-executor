"""Execution engine versioning.

Different strategy modules may have functions with different signatures.
Versioning strategy modules allows us to add and remove arguments
without breaking backwards compatibility.
"""

from typing import TypeAlias

#: Different strategy module layouts.
#:
#: No changelog available yet.
#:
SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5")

#: Data type for engine versions.
#:
#: See :py:data:`SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS`
#:
TradingStrategyEngineVersion: TypeAlias = str
