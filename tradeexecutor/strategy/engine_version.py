"""Execution engine versioning."""

from typing import TypeAlias

#: Different strategy module layouts.
#:
#: No changelog available yet.
#:
SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS = ("0.1", "0.2", "0.3")

#: Data type for engine versions.
#:
#: See :py:data:`SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS`
#:
TradingStrategyEngineVersion: TypeAlias = str
