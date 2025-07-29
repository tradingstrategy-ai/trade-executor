"""Execution engine versioning.

Different strategy modules may have functions with different signatures.
Versioning strategy modules allows us to add and remove arguments
without breaking backwards compatibility.

To compare versions:

.. code-block:: python

    from packaging import version

    engine_version = run_description.trading_strategy_engine_version
    if engine_version:
        if version.parse(engine_version) >= version.parse("0.5"):
            parameters = run_description.runner.parameters
            assert "required_history_period" in parameters, f"Strategy lacks Parameters.required_history_period. We have {parameters}"


"""

from typing import TypeAlias

#: Different strategy module layouts.
#:
#: No changelog available yet.
#:
SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6")

#: Data type for engine versions.
#:
#: See :py:data:`SUPPORTED_TRADING_STRATEGY_ENGINE_VERSIONS`
#:
TradingStrategyEngineVersion: TypeAlias = str
