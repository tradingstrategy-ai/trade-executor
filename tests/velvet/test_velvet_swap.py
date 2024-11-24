"""Execute trades using Velvet vault and Enso."""
import datetime
import secrets
from decimal import Decimal

import pytest
import flaky

from eth_defi.enzyme.integration_manager import IntegrationManagerActionId
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_typing import HexAddress

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3
from web3.contract import Contract

from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor

from tradeexecutor.ethereum.enzyme.tx import EnzymeTransactionBuilder
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.testing.ethereumtrader_uniswap_v2 import UniswapV2TestTrader



@pytest.fixture()
def routing_model() -> VelvetEnsoRouting:
    return VelvetEnsoRouting()


@pytest.fixture()
def pricing_model(
    web3,
    uniswap_v2,
    pair_universe: PandasPairUniverse,
    routing_model
) -> UniswapV2LivePricing:
    pricing_model = UniswapV2LivePricing(
        web3,
        pair_universe,
        routing_model,
    )
    return pricing_model
