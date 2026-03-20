import os
from types import SimpleNamespace
from unittest.mock import patch

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
