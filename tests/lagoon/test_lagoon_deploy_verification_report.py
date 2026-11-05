"""Test the Etherscan status report after a Lagoon deployment.

Deployment submits the trading module to nobody, so the status must be read from
Etherscan. The API key is in the request URL and must not reach the logs.
"""

import logging
from typing import Any

import pytest
import requests
from pytest_mock import MockerFixture

import tradeexecutor.ethereum.lagoon.deploy_report as report_module
from tradeexecutor.ethereum.lagoon.deploy_report import (
    ContractVerificationStatus,
    add_module_verification_report,
    fetch_etherscan_verification_status,
)

MODULE = "0xf85523b94Ee7268bA26B0a79682Aec71EB637855"
API_KEY = "ETHERSCAN-KEY-DO-NOT-LOG"


def create_response(mocker: MockerFixture, payload: Any, status_code: int = 200) -> Any:
    """Create a fake Etherscan response, in the shape read from the live API on 2026-11-05."""
    response = mocker.Mock(status_code=status_code)
    response.json.return_value = payload
    return response


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "1", "result": [{"SourceCode": "contract X {}"}]}, ContractVerificationStatus.verified),
        ({"status": "1", "result": [{"SourceCode": ""}]}, ContractVerificationStatus.not_verified),
        ({"status": "0", "message": "NOTOK", "result": "Max rate limit reached"}, ContractVerificationStatus.unknown),
        ({"status": "1", "result": []}, ContractVerificationStatus.unknown),
    ],
)
def test_verification_status_is_read_from_the_response(
    mocker: MockerFixture,
    payload: dict[str, Any],
    expected: ContractVerificationStatus,
) -> None:
    """Only a valid result row decides the status, as API errors arrive as HTTP 200.

    The HTTP call is mocked, because the live response needs an API key.

    1. Fake an Etherscan response.
    2. Read the status.
    3. Verify it, and that the request cannot hang.
    """
    # 1. Fake an Etherscan response.
    get = mocker.patch.object(report_module.requests, "get", return_value=create_response(mocker, payload))

    # 2. Read the status.
    status = fetch_etherscan_verification_status(1, MODULE, API_KEY)

    # 3. Verify it, and that the request cannot hang.
    assert status == expected
    assert get.call_args.kwargs["timeout"] > 0


def test_failed_check_keeps_the_deployment_and_the_key_safe(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An Etherscan failure neither raises nor logs the API key.

    1. Make the request time out with the URL in its message, as requests does.
    2. Add the status to a report.
    3. Verify the report says unknown and the logs do not hold the key.
    """
    # 1. Make the request time out with the URL in its message, as requests does.
    error = requests.Timeout(f"https://api.etherscan.io/v2/api?apikey={API_KEY}")
    mocker.patch.object(report_module.requests, "get", side_effect=error)

    # 2. Add the status to a report.
    with caplog.at_level(logging.INFO):
        report = add_module_verification_report("# Report", {1: MODULE}, API_KEY)

    # 3. Verify the report says unknown and the logs do not hold the key.
    assert f"`{MODULE}` on chain 1: unknown" in report
    assert API_KEY not in caplog.text
    assert API_KEY not in report


def test_unverified_module_is_reported(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unverified module is a warning, because operators assume the key covered it.

    1. Fake Etherscan reporting no source.
    2. Add the status to a report.
    3. Verify the report and the warning.
    """
    # 1. Fake Etherscan reporting no source.
    response = create_response(mocker, {"status": "1", "result": [{"SourceCode": ""}]})
    mocker.patch.object(report_module.requests, "get", return_value=response)

    # 2. Add the status to a report.
    with caplog.at_level(logging.WARNING):
        report = add_module_verification_report("# Report", {1: MODULE}, API_KEY)

    # 3. Verify the report and the warning.
    assert report.startswith("# Report")
    assert f"`{MODULE}` on chain 1: not_verified" in report
    assert "verify them by hand" in caplog.text
