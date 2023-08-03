import os
from unittest.mock import patch

from eth_defi.gas import GasPriceMethod
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
