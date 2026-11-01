"""Lagoon safe block selection for RPCs on fast chains."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hexbytes import HexBytes
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.lagoon import vault as lagoon_vault
from tradeexecutor.ethereum.vault import settlement_retry


@pytest.mark.parametrize("chain_id,provider_delay,confirmations", [
    (ChainId.ethereum.value, 0, None),
    (ChainId.ethereum.value, 4, 4),
    (ChainId.base.value, 4, 16),
    (ChainId.base.value, 20, 20),
])
def test_vault_claim_wait_covers_treasury_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    chain_id: int,
    provider_delay: int,
    confirmations: int | None,
) -> None:
    """A normal claim wait reaches the accounting horizon without losing a cycle.

    1. Configure Ethereum or a fast chain with its existing provider delay.
    2. Confirm a claim through the shared receipt wrapper.
    3. Keep the time-based default and ensure the treasury snapshot includes the claim.
    """
    # 1. Configure a tip at the required horizon after a claim mined at block 100.
    web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=chain_id, block_number=100 + (confirmations or 0)), block_tip_latency=provider_delay)
    receipt = {"blockNumber": 100}
    wait = Mock(return_value=receipt)
    monkeypatch.setattr(settlement_retry, "wait_for_transaction_receipt_robust", wait)
    monkeypatch.setattr(lagoon_vault, "get_block_tip_latency", lambda _web3: provider_delay)
    monkeypatch.setattr(lagoon_vault, "get_almost_latest_block_number", lambda _web3: _web3.eth.block_number - provider_delay)

    # 2. Use the production wrapper rather than introducing another wait loop.
    assert settlement_retry._wait_for_settlement_tx_receipt(web3, HexBytes("0x" + "11" * 32)) is receipt

    # 3. None retains the SDK's time-based wait; every selected snapshot is safe.
    assert wait.call_args.kwargs["confirmation_block_count"] == confirmations
    model = SimpleNamespace(web3=web3, anvil=False, unit_testing=False)
    assert lagoon_vault.LagoonVaultSyncModel.get_safe_latest_block(model) >= receipt["blockNumber"]


@pytest.mark.parametrize(
    "chain_id", [ChainId.arbitrum.value, ChainId.base.value, 999999]
)
def test_fast_chain_lagoon_uses_sixteen_block_buffer(
    monkeypatch: pytest.MonkeyPatch, chain_id: int
):
    """A fast chain uses at least 16 blocks even when the provider's own delay is shorter.

    1. Configure a fast chain with a four-block provider delay.
    2. Select the Lagoon safe block.
    3. Confirm a later RPC read six blocks behind the first remains ahead of it.
    """
    # 1. Replace provider detection so this test covers the Lagoon policy itself.
    monkeypatch.setattr(lagoon_vault, "get_block_tip_latency", lambda web3: 4)
    web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=chain_id, block_number=712))
    model = SimpleNamespace(web3=web3, anvil=False, unit_testing=False)

    # 2. Select the Lagoon safe block.
    safe_block = lagoon_vault.LagoonVaultSyncModel.get_safe_latest_block(model)

    # 3. The production failure's later tip of 706 is still ahead of block 696.
    assert lagoon_vault.is_fast_chain(chain_id)
    assert safe_block == 696
    assert 706 >= safe_block


def test_fast_chain_lagoon_keeps_larger_provider_buffer(
    monkeypatch: pytest.MonkeyPatch,
):
    """A provider requiring more than 16 blocks retains its stronger delay.

    1. Configure a fast chain with a 20-block provider delay.
    2. Select the Lagoon safe block.
    3. Confirm the provider's 20-block delay is used.
    """
    # 1. Replace provider detection to exercise its larger-delay branch.
    monkeypatch.setattr(lagoon_vault, "get_block_tip_latency", lambda web3: 20)
    web3 = SimpleNamespace(
        eth=SimpleNamespace(chain_id=ChainId.arbitrum.value, block_number=712)
    )
    model = SimpleNamespace(web3=web3, anvil=False, unit_testing=False)

    # 2. Select the Lagoon safe block.
    safe_block = lagoon_vault.LagoonVaultSyncModel.get_safe_latest_block(model)

    # 3. Keep the stronger provider delay.
    assert safe_block == 692


def test_ethereum_lagoon_keeps_existing_safe_block(monkeypatch: pytest.MonkeyPatch):
    """Ethereum mainnet retains the existing provider-based safe block.

    1. Configure Ethereum and the existing safe block helper.
    2. Select the Lagoon safe block.
    3. Confirm no fast-chain buffer is applied.
    """
    # 1. Stub the provider helper because its behaviour is outside this change.
    monkeypatch.setattr(
        lagoon_vault, "get_almost_latest_block_number", lambda web3: 708
    )
    web3 = SimpleNamespace(
        eth=SimpleNamespace(chain_id=ChainId.ethereum.value, block_number=712)
    )
    model = SimpleNamespace(web3=web3, anvil=False, unit_testing=False)

    # 2. Select the Lagoon safe block.
    safe_block = lagoon_vault.LagoonVaultSyncModel.get_safe_latest_block(model)

    # 3. Keep the existing Ethereum calculation.
    assert not lagoon_vault.is_fast_chain(ChainId.ethereum.value)
    assert safe_block == 708
