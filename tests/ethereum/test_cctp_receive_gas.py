"""Regression tests for CCTP receiveMessage gas-limit estimation.

The destination-chain ``receiveMessage`` mint was previously signed with a
hardcoded 200_000 gas limit — less than the 300_000 used for the lighter
``depositForBurn`` burn. Because ``receiveMessage`` also verifies Circle's
attestation signature and performs a (possibly cold) USDC mint, 200_000 ran
out of gas and reverted on-chain, stranding the burn ``cctp_in_transit`` with
no funds minted on the destination chain.

``estimate_receive_message_gas()`` now estimates against the live node with a
2x safety buffer and falls back to a generous fixed limit when estimation
fails (e.g. a transient RPC error like the one that originally masked the bug).
"""

from tradeexecutor.ethereum.cctp.routing import (
    CCTP_RECEIVE_GAS_FALLBACK,
    estimate_receive_message_gas,
)


class _FakeReceiveFn:
    """Stand-in for a bound ``receiveMessage`` contract function."""

    def __init__(self, *, estimate: int | None = None, raises: bool = False):
        self._estimate = estimate
        self._raises = raises

    def estimate_gas(self, tx: dict) -> int:
        if self._raises:
            raise ValueError("execution reverted / node lacked block data")
        return self._estimate


def test_estimate_receive_message_gas_happy_and_fallback():
    """Gas estimation buffers a live estimate and falls back when it fails.

    1. A successful estimate is doubled to absorb cold-vs-warm storage drift.
    2. A small estimate is floored at 300_000 so it never drops below the burn
       gas limit again (the original bug).
    3. When estimation raises (transient RPC error), the generous fixed
       fallback is used instead of failing the bridge.
    """

    # 1. A realistic estimate is doubled for safety headroom
    assert estimate_receive_message_gas(_FakeReceiveFn(estimate=250_000), "0xabc") == 500_000

    # 2. A tiny estimate is floored so we never undercut the burn's 300_000
    assert estimate_receive_message_gas(_FakeReceiveFn(estimate=120_000), "0xabc") == 300_000

    # 3. Estimation failure falls back to the fixed generous limit
    assert estimate_receive_message_gas(_FakeReceiveFn(raises=True), "0xabc") == CCTP_RECEIVE_GAS_FALLBACK
    assert CCTP_RECEIVE_GAS_FALLBACK >= 1_000_000
