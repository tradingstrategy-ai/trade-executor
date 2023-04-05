"""Pytest helpers.

"""

from typing import Dict

from pytest import StashKey, CollectReport, FixtureRequest


#: See https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
phase_report_key = StashKey[Dict[str, CollectReport]]()


def is_failed_test(request: FixtureRequest) -> bool:
    """Check if the underlying test failed withint pytest fixture.

    `See details here <https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures>`__.
    """
    report = request.node.stash[phase_report_key]
    if report["setup"].failed:
        return True
    elif ("call" not in report) or report["call"].failed:
        return True

    return False