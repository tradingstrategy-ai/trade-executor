"""Test source chain balance checks for distribute-gas-funds."""

from unittest import mock

import pytest

from eth_defi.hotwallet import HotWallet

from tradeexecutor.cli.commands.distribute_gas_funds import assert_source_chain_has_balance


def test_assert_source_chain_has_balance_fails_with_strategy_parameter_id() -> None:
    """Zero source chain gas balance fails early with strategy parameter id context.

    1. Create a hot wallet and a mocked source chain Web3 connection.
    2. Mock the source chain native balance as zero because bridge transactions need source chain gas.
    3. Assert the pre-flight check fails before any distribution flow can continue.
    """
    wallet = HotWallet.from_private_key("0x" + "1" * 64)
    source_web3 = mock.Mock()
    source_web3.eth.get_balance.return_value = 0

    # 1. Create a hot wallet and a mocked source chain Web3 connection.
    # 2. Mock the source chain native balance as zero because bridge transactions need source chain gas.
    # 3. Assert the pre-flight check fails before any distribution flow can continue.
    with pytest.raises(AssertionError) as exc_info:
        assert_source_chain_has_balance(
            source_web3=source_web3,
            wallet=wallet,
            strategy_parameter_id="xchain-master-vault",
            source_chain_name="Ethereum",
            source_chain_id=1,
        )

    message = str(exc_info.value)
    assert "xchain-master-vault" in message
    assert "Ethereum" in message
    assert "chain_id=1" in message
    assert wallet.address in message
    source_web3.eth.get_balance.assert_called_once_with(wallet.address)
