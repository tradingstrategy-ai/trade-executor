import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.gas import GasPriceMethod
from eth_defi.hyperliquid.block import HYPEREVM_BIG_BLOCK_GAS_LIMIT
from eth_defi.provider.multi_provider import MultiProviderWeb3
from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.ethereum.web3config import Web3Config
from tradingstrategy.chain import ChainId


def test_configure_web3():
    """Test Web3 configuration"""

    setup_custom_log_levels()

    # Non-sense test config
    environment = {
        "JSON_RPC_POLYGON": "mev+https://polygon-rpc.com https://bsc-dataseed2.bnbchain.org",
        "JSON_RPC_BINANCE": "https://bsc-dataseed2.bnbchain.org"
    }

    config = Web3Config.setup_from_environment(GasPriceMethod.legacy, **environment)

    assert len(config.connections) == 2
    assert isinstance(config.connections[ChainId.polygon], MultiProviderWeb3)
    assert isinstance(config.connections[ChainId.bsc], MultiProviderWeb3)

    polygon_web3 = config.connections[ChainId.polygon]
    assert polygon_web3.get_active_transact_provider().endpoint_uri == "https://polygon-rpc.com"
    assert polygon_web3.get_active_call_provider().endpoint_uri == "https://bsc-dataseed2.bnbchain.org"


def test_configure_web3_simulate_uses_custom_http_timeout():
    setup_custom_log_levels()
    captured = {}

    class FakeEth:
        chain_id = ChainId.anvil.value

        def set_gas_price_strategy(self, strategy):
            captured["gas_strategy"] = strategy

    fake_web3 = SimpleNamespace(eth=FakeEth())

    with patch("tradeexecutor.ethereum.web3config.launch_anvil") as launch_anvil_mock:
        launch_anvil_mock.return_value = SimpleNamespace(json_rpc_url="http://127.0.0.1:8545")

        with patch("tradeexecutor.ethereum.web3config.create_multi_provider_web3") as create_web3_mock:
            create_web3_mock.return_value = fake_web3

            config = Web3Config.setup_from_environment(
                GasPriceMethod.legacy,
                simulate=True,
                simulate_http_timeout=(3.0, 90.0),
                json_rpc_base="https://example-rpc.invalid",
            )

    assert config.connections[ChainId.base] is fake_web3
    assert create_web3_mock.call_args.kwargs["default_http_timeout"] == (3.0, 90.0)


def test_configure_web3_simulate_hyperliquid_uses_big_block_gas_limit():
    """Test Hyperliquid simulate mode launches Anvil with the large block gas limit.

    1. Set up a simulated Hyperliquid RPC configuration with patched Anvil and Web3 factories.
    2. Build the Web3 config through the normal environment loader.
    3. Confirm the Anvil launcher receives the HyperEVM large-block gas limit override.
    """
    setup_custom_log_levels()

    class FakeEth:
        chain_id = ChainId.anvil.value

        def set_gas_price_strategy(self, strategy):
            return None

    fake_web3 = SimpleNamespace(eth=FakeEth())

    # 1. Set up a simulated Hyperliquid RPC configuration with patched Anvil and Web3 factories.
    with patch("tradeexecutor.ethereum.web3config.launch_anvil") as launch_anvil_mock:
        launch_anvil_mock.return_value = SimpleNamespace(json_rpc_url="http://127.0.0.1:8545")

        with patch("tradeexecutor.ethereum.web3config.create_multi_provider_web3") as create_web3_mock:
            create_web3_mock.return_value = fake_web3

            # 2. Build the Web3 config through the normal environment loader.
            config = Web3Config.setup_from_environment(
                GasPriceMethod.legacy,
                simulate=True,
                json_rpc_hyperliquid="https://example-hyperliquid-rpc.invalid",
            )

    # 3. Confirm the Anvil launcher receives the HyperEVM large-block gas limit override.
    assert config.connections[ChainId.hyperliquid] is fake_web3
    assert launch_anvil_mock.call_args.kwargs["gas_limit"] == HYPEREVM_BIG_BLOCK_GAS_LIMIT


def test_simulated_multichain_setup_failure_closes_every_started_anvil() -> None:
    """A partial multichain setup cannot leak Anvil processes.

    1. Return two simulated chain connections and fail while starting the third.
    2. Make the first Anvil shutdown fail to exercise best-effort cleanup.
    3. Verify both started forks receive a bounded shutdown and the original setup error survives.
    """
    first_anvil = MagicMock()
    first_anvil.process.pid = 1001
    first_anvil.close.side_effect = RuntimeError("first Anvil would not stop")
    second_anvil = MagicMock()
    second_anvil.process.pid = 1002
    first_web3 = SimpleNamespace(anvil=first_anvil)
    second_web3 = SimpleNamespace(anvil=second_anvil)
    setup_error = RuntimeError("third Anvil did not start")

    # 1. Return two simulated chain connections and fail while starting the third.
    with patch.object(
        Web3Config,
        "create_web3",
        side_effect=[first_web3, second_web3, setup_error],
    ):
        # 2. Make the first Anvil shutdown fail to exercise best-effort cleanup.
        with pytest.raises(RuntimeError, match="third Anvil did not start") as raised:
            Web3Config.setup_from_environment(
                GasPriceMethod.legacy,
                simulate=True,
                json_rpc_ethereum="https://ethereum.example",
                json_rpc_avalanche="https://avalanche.example",
                json_rpc_polygon="https://polygon.example",
            )

    # 3. Verify both started forks receive a bounded shutdown and the original setup error survives.
    assert raised.value is setup_error
    first_anvil.close.assert_called_once_with(log_level=logging.ERROR, block_timeout=5)
    second_anvil.close.assert_called_once_with(log_level=logging.ERROR, block_timeout=5)


def test_check_default_chain_id_live_strict_vs_unit_testing():
    """check_default_chain_id() only tolerates a test-chain node in test/fork mode.

    A node reporting a test chain id (e.g. plain Anvil 31337) may stand in for a
    real chain in unit tests, but a live command pointing a ``JSON_RPC_*`` at such a
    node is a misconfiguration that must fail fast.

    1. A live command (unit_testing=False) where the node reports Anvil 31337 but the
       strategy expects Arbitrum must raise.
    2. The same configuration is tolerated when unit_testing=True.
    """

    # Fake web3 reporting the plain Anvil chain id, mapped as if it were Arbitrum.
    fake_web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=ChainId.anvil.value))

    config = Web3Config()
    config.connections[ChainId.arbitrum] = fake_web3
    config.set_default_chain(ChainId.arbitrum)

    # 1. Live command must fail fast on the chain mismatch.
    config.unit_testing = False
    with pytest.raises(AssertionError):
        config.check_default_chain_id()

    # 2. Under unit testing the test-chain node is tolerated.
    config.unit_testing = True
    config.check_default_chain_id()
