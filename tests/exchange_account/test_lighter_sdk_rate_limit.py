"""Test Lighter rate limiting with the real SDK against a local HTTP server.

The fake sequencer in :py:mod:`tradeexecutor.testing.lighter_core` replaces the
SDK client, so it cannot see the transport. A four-sub-account mainnet deployment
stopped on 2026-11-04 on Lighter's limit of 2 ``L2CreateSubAccount`` requests per
60 seconds. Here a real ``SignerClient`` signs real transactions and a local server
enforces the same limit on a shorter window.
"""

import asyncio
import dataclasses
import json
import time
from pathlib import Path

import eth_defi.lighter.sub_account as sub_account_module
import lighter
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from eth_defi.lighter.api_key import generate_lighter_api_key
from eth_defi.lighter.sdk import create_lighter_api_client
from eth_defi.lighter.session import LIGHTER_SDK_DEFAULT_POLICY, LIGHTER_SDK_SLOW_TRANSACTION_POLICY, LIGHTER_SEND_TX_PATH
from eth_defi.lighter.sub_account import _submit_sub_account_creations
from eth_defi.lighter.testing import find_lighter_session_bypasses

MASTER_ACCOUNT_INDEX = 766884
API_KEY_INDEX = 4
FIRST_NONCE = 10
SUB_ACCOUNTS = 4
#: Half a second stands in for Lighter's 60-second window.
RATE_LIMIT_WINDOW = 0.5
RATE_LIMIT_REQUESTS = 2
REPOSITORY_ROOT = Path(__file__).parents[2]


def create_sequencer() -> tuple[web.Application, list[tuple[int, str]]]:
    """Create a server enforcing the sub-account limit and recording every ``sendTx``.

    :return:
        Application and the ``(status, tx_info)`` of each ``sendTx`` request.
    """
    sends: list[tuple[int, str]] = []
    accepted_at: list[float] = []
    dropped_read = [False]

    async def next_nonce(request: web.Request) -> web.StreamResponse:
        """Break the first read after its headers, then answer normally."""
        if not dropped_read[0]:
            dropped_read[0] = True
            response = web.StreamResponse(headers={"Content-Length": "100"})
            await response.prepare(request)
            await response.write(b'{"code"')
            request.transport.close()
            return response
        return web.json_response({"code": 200, "nonce": FIRST_NONCE})

    async def send_tx(request: web.Request) -> web.Response:
        """Refuse a transaction above the limit, as Lighter's API server does.

        The header exercises the session's read of it on the SDK's real response.
        """
        form = await request.post()
        tx_info = form["tx_info"]
        now = time.monotonic()
        accepted_at[:] = [moment for moment in accepted_at if now - moment < RATE_LIMIT_WINDOW]
        if len(accepted_at) >= RATE_LIMIT_REQUESTS:
            sends.append((429, tx_info))
            return web.json_response({"code": 23000, "message": "Too Many Requests!: L1Address ratelimit reached"}, status=429, headers={"Retry-After": "0"})
        accepted_at.append(now)
        sends.append((200, tx_info))
        return web.json_response({"code": 200, "tx_hash": f"0x{len(accepted_at):064x}", "predicted_execution_time_ms": 0})

    application = web.Application()
    application.router.add_get("/api/v1/nextNonce", next_nonce)
    application.router.add_post(LIGHTER_SEND_TX_PATH, send_tx)
    return application, sends


@pytest.mark.timeout(60)
def test_four_sub_accounts_survive_the_real_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """All four creations land when Lighter refuses the third, without a nonce gap.

    1. Serve the limit locally and shorten the backoff to match its window.
    2. Create four sub-accounts through eth-defi's real SDK path.
    3. Assert the limit was hit and each refused transaction was resent unchanged.
    4. Assert exactly four transactions were accepted, with consecutive nonces.
    5. Assert a signed transaction outlives the 300-second deadline it can be resent for.
    """
    # 1. Serve the limit locally and shorten the backoff to match its window.
    application, sends = create_sequencer()
    key = generate_lighter_api_key(API_KEY_INDEX)
    policy = dataclasses.replace(LIGHTER_SDK_DEFAULT_POLICY, initial_delay=0.05, max_delay=0.4, read_deadline=5.0, transaction_deadline=5.0)
    monkeypatch.setattr(sub_account_module, "LIGHTER_SDK_SLOW_TRANSACTION_POLICY", policy)

    async def create_sub_accounts() -> None:
        server = TestServer(application)
        await server.start_server()
        try:
            await _submit_sub_account_creations(MASTER_ACCOUNT_INDEX, API_KEY_INDEX, key.private_key, SUB_ACCOUNTS, str(server.make_url("")).rstrip("/"))
        finally:
            await server.close()

    # 2. Create four sub-accounts through eth-defi's real SDK path.
    asyncio.run(create_sub_accounts())

    # 3. Assert the limit was hit and each refused transaction was resent unchanged.
    refused = [tx_info for status, tx_info in sends if status == 429]
    accepted = [tx_info for status, tx_info in sends if status == 200]
    assert refused
    assert set(refused) <= set(accepted)

    # 4. Assert exactly four transactions were accepted, with consecutive nonces.
    assert len(accepted) == SUB_ACCOUNTS
    assert [json.loads(tx_info)["Nonce"] for tx_info in accepted] == list(range(FIRST_NONCE, FIRST_NONCE + SUB_ACCOUNTS))

    # 5. Assert a signed transaction outlives the 300-second deadline it can be resent for.
    expires_in = min(json.loads(tx_info)["ExpiredAt"] for tx_info in accepted) / 1000 - time.time()
    assert expires_in > LIGHTER_SDK_SLOW_TRANSACTION_POLICY.transaction_deadline


@pytest.mark.timeout(60)
def test_sdk_read_survives_truncated_body() -> None:
    """The SDK deserialises a good response after a read broke part-way, and closes cleanly.

    1. Serve a read that drops the connection after its headers, then one that works.
    2. Read through the protected API client with the SDK's generated method.
    3. Assert the second response was deserialised.
    """
    # 1. Serve a read that drops the connection after its headers, then one that works.
    application, _sends = create_sequencer()
    policy = dataclasses.replace(LIGHTER_SDK_DEFAULT_POLICY, initial_delay=0.01, max_delay=0.02)

    async def read_nonce() -> int:
        server = TestServer(application)
        await server.start_server()
        api_client = create_lighter_api_client(str(server.make_url("")).rstrip("/"), retry_policy=policy)
        try:
            # 2. Read through the protected API client with the SDK's generated method.
            response = await lighter.TransactionApi(api_client).next_nonce(account_index=MASTER_ACCOUNT_INDEX, api_key_index=API_KEY_INDEX)
            return response.nonce
        finally:
            await api_client.close()
            await server.close()

    # 3. Assert the second response was deserialised.
    assert asyncio.run(read_nonce()) == FIRST_NONCE


@pytest.mark.timeout(30)
def test_no_lighter_code_bypasses_eth_defi_sessions() -> None:
    """Executor code cannot reach Lighter without backoff.

    1. Collect every executor module and script for Lighter.
    2. Assert none of them skips eth-defi's sessions.
    """
    # 1. Collect every executor module and script for Lighter.
    paths = [path for root in ("tradeexecutor", "scripts") for path in (REPOSITORY_ROOT / root).rglob("*lighter*.py")]
    paths += list((REPOSITORY_ROOT / "tradeexecutor" / "ethereum" / "lighter").glob("*.py"))
    assert len(paths) > 10

    # 2. Assert none of them skips eth-defi's sessions.
    assert find_lighter_session_bypasses(paths) == []
