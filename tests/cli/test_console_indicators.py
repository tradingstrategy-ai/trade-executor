"""Tests for console indicator fallback behaviour."""

import datetime
import logging
from types import SimpleNamespace

import pytest

from tradeexecutor.cli.commands import console as console_module


class _FakeIndicatorSet:
    """Minimal indicator set stub for console indicator fallback tests."""

    def generate_combinations(self, universe) -> list[str]:
        return ["failing_indicator"]


class _FakeUniverse:
    """Minimal universe stub exposing the cache key used by the console helper."""

    def get_cache_key(self) -> str:
        return "test-universe"


def test_calculate_console_indicators_warns_and_disables_indicators(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Check console starts without indicators when indicator calculation fails.

    The mocked ``calculate_indicators`` fails after storage and indicator set
    setup to verify partially initialised indicator objects are not exposed in
    console bindings.

    1. Mock console indicator preparation to create a fake indicator set.
    2. Mock indicator calculation to fail like a live data lookup issue.
    3. Verify all indicator bindings are disabled and a warning is logged.
    """

    mod = SimpleNamespace(
        create_indicators=lambda *args, **kwargs: None,
        parameters=SimpleNamespace(),
    )
    logger = logging.getLogger("test-console-indicators")

    # 1. Mock console indicator preparation to create a fake indicator set.
    monkeypatch.setattr(console_module, "prepare_indicators", lambda *args, **kwargs: _FakeIndicatorSet())
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # 2. Mock indicator calculation to fail like a live data lookup issue.
    def fail_calculate_indicators(*args, **kwargs):
        raise RuntimeError("missing supporting pair")

    monkeypatch.setattr(console_module, "calculate_indicators", fail_calculate_indicators)

    with caplog.at_level(logging.WARNING, logger="test-console-indicators"):
        indicator_storage, indicator_set, indicator_result_map, indicators = console_module.calculate_console_indicators(
            mod=mod,
            universe=_FakeUniverse(),
            execution_context=None,
            cycle_timestamp=datetime.datetime(2026, 6, 19),
            max_workers=1,
            logger=logger,
        )

    # 3. Verify all indicator bindings are disabled and a warning is logged.
    assert indicator_storage is None
    assert indicator_set is None
    assert indicator_result_map is None
    assert indicators is None
    assert "Could not calculate real-time indicators for console; continuing without indicators" in caplog.text
