"""Lagoon safe block selection for RPCs on fast chains."""

from types import SimpleNamespace

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.lagoon import vault as lagoon_vault


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
