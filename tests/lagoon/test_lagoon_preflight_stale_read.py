"""Lagoon settlement preflight retries for a node lagging behind the NAV post.

Each simulation raises the JSON-RPC envelope ``FallbackProvider`` raised in the
hyper-ai stop on 2026-11-03, so this also covers revert data extraction.
"""
import logging
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

import pytest
from eth_abi import encode
from eth_defi.provider.fallback import ExtraValueError
from web3 import HTTPProvider, Web3

from tradeexecutor.ethereum.lagoon.vault import (
    LAGOON_SETTLEMENT_WINDOW_LIMIT_EXCEEDED_SELECTOR,
    LagoonVaultSyncModel,
)


NEW_TOTAL_ASSETS_MISSING = Web3.keccak(text="NewTotalAssetsMissing()")[:4]
WRONG_NEW_TOTAL_ASSETS = Web3.keccak(text="WrongNewTotalAssets()")[:4]
#: A guarded module revert that says nothing about the posted NAV.
UNRELATED_REVERT = Web3.keccak(text="SomethingElse()")[:4]
#: The executor's Lighter session schedule, which sums to the 600 second budget.
BACKOFF_SCHEDULE = [0, 1, 2, 4, 8, 16, 32, 64, 120, 120, 120, 113]


def revert(data: bytes) -> ExtraValueError:
    """Build the provider exception for a reverted ``eth_call``."""
    return ExtraValueError({"code": 3, "message": "execution reverted", "data": "0x" + data.hex()})


def preflight(simulation_results: list, guarded: bool = True) -> tuple[object, mock.Mock]:
    """Run the production preflight path against scripted simulation results.

    :param simulation_results:
        Passed as ``eth_call`` side effects: an exception raises, a value returns.

    :param guarded:
        Whether the module has GuardV0 settlement safety.

    :return:
        Preflight result and the ``eth_call`` mock.
    """
    call = mock.Mock(side_effect=simulation_results)
    token = SimpleNamespace(convert_to_raw=lambda amount: int(amount))
    sync_model = SimpleNamespace(
        vault=SimpleNamespace(
            get_flow_manager=lambda: SimpleNamespace(
                fetch_pending_deposit=lambda block: Decimal(0),
                fetch_pending_redemption=lambda block: Decimal(0),
            ),
            underlying_token=token,
            share_token=token,
            trading_strategy_module_address="0x0000000000000000000000000000000000000001",
        ),
        web3=SimpleNamespace(
            eth=SimpleNamespace(call=call),
            provider=HTTPProvider("http://transact.example"),
            get_active_call_provider=lambda: HTTPProvider("http://read.example"),
        ),
        anvil=False,
        unit_testing=False,
        hot_wallet=SimpleNamespace(address="0x0000000000000000000000000000000000000002"),
        _has_lagoon_settlement_safety=lambda: guarded,
    )
    settle_func = SimpleNamespace(
        address="0x0000000000000000000000000000000000000003",
        _encode_transaction_data=lambda: "0xd24ca58a",
    )
    return LagoonVaultSyncModel._preflight_lagoon_settlement(sync_model, settle_func), call


@pytest.fixture()
def sleeps():
    """Replace the clock so waits are recorded instead of spent."""
    recorded: list[float] = []
    fake_time = SimpleNamespace(
        monotonic=lambda: float(sum(recorded)),
        sleep=recorded.append,
    )
    with mock.patch("tradeexecutor.ethereum.lagoon.vault.time", fake_time):
        yield recorded


@pytest.mark.parametrize(
    "selector, failures",
    [
        (NEW_TOTAL_ASSETS_MISSING, 2),
        (WRONG_NEW_TOTAL_ASSETS, 2),
        (NEW_TOTAL_ASSETS_MISSING, len(BACKOFF_SCHEDULE)),
    ],
)
def test_stale_read_settles_once_the_node_catches_up(
    selector: bytes,
    failures: int,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
):
    """A node that catches up within the budget allows settlement.

    1. Fail the first simulations with a stale NAV selector.
    2. Let the next simulation pass, which is the last permitted attempt in one case.
    3. Verify settlement is allowed after the matching backoff sleeps.
    4. Verify the log names the read provider, not the transaction endpoint.
    """
    # 1. and 2. Script the lagging node.
    with caplog.at_level(logging.WARNING):
        result, call = preflight([revert(selector) for _ in range(failures)] + [b""])

    # 3. Settlement goes ahead.
    assert result.should_settle is True
    assert call.call_count == failures + 1
    assert sleeps == BACKOFF_SCHEDULE[:failures]

    # 4. An incident log must identify the node that served the simulation.
    assert "provider=read.example" in caplog.text
    assert "transact.example" not in caplog.text


def test_stale_read_exhaustion_stops_the_caller(sleeps: list[float]):
    """A NAV the node never sees fails as an unexpected guarded revert.

    1. Fail every simulation with a stale NAV selector.
    2. Verify the provider exception reaches the caller, so nothing is settled.
    3. Verify the backoff schedule and that it stops at the time budget.
    """
    # 1. The selector may be genuine, for example after a direct Safe settlement.
    failures = [revert(NEW_TOTAL_ASSETS_MISSING) for _ in BACKOFF_SCHEDULE] + [revert(NEW_TOTAL_ASSETS_MISSING)]

    # 2. The last provider exception is raised unchanged.
    with pytest.raises(ExtraValueError) as exc_info:
        preflight(failures)
    assert exc_info.value is failures[-1]

    # 3. All scripted calls were used, and one more would raise StopIteration.
    assert sleeps == BACKOFF_SCHEDULE


def test_guard_limit_after_stale_read_keeps_its_deferral(sleeps: list[float]):
    """A node that catches up can report a real window-budget breach.

    1. Fail the first simulation with a stale NAV selector.
    2. Fail the second with the GuardV0 window limit.
    3. Verify the existing manual settlement deferral applies.
    """
    # 1. and 2. Script the catch-up with used, actual and maximum raw amounts.
    limit_data = LAGOON_SETTLEMENT_WINDOW_LIMIT_EXCEEDED_SELECTOR + encode(["uint256", "uint256", "uint256"], [1, 2, 3])
    result, call = preflight([revert(WRONG_NEW_TOTAL_ASSETS), revert(limit_data)])

    # 3. The queue stays pending for Safe governance.
    assert result.should_settle is False
    assert result.manual_settlement_required is True
    assert (result.already_used_raw, result.actual_amount_raw, result.max_amount_raw) == (1, 2, 3)
    assert call.call_count == 2
    assert sleeps == BACKOFF_SCHEDULE[:1]


def test_unrelated_guarded_revert_is_not_retried(sleeps: list[float]):
    """Other guarded module reverts keep failing fast.

    1. Fail the simulation with an unrelated selector.
    2. Verify it raises at once without waiting.
    """
    # 1. Script a single failure: a retry would raise StopIteration instead.
    # 2. The provider exception reaches the caller.
    with pytest.raises(ExtraValueError):
        preflight([revert(UNRELATED_REVERT)])
    assert sleeps == []


def test_module_without_settlement_safety_keeps_its_warning_path(sleeps: list[float]):
    """Older modules defer a failed empty-queue simulation without retrying.

    1. Fail the simulation on a module without settlement safety.
    2. Verify the existing logged deferral applies after one attempt.
    """
    # 1. Script one failure.
    result, call = preflight([revert(NEW_TOTAL_ASSETS_MISSING)], guarded=False)

    # 2. No settlement, no retry.
    assert result.should_settle is False
    assert call.call_count == 1
    assert sleeps == []
