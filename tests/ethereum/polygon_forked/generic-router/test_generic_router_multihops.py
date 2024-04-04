"""Test live routing of combined Quickswap spot position, trading through several hops."""
import datetime
import os
import shutil
from decimal import Decimal

import pytest
import pandas as pd
from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.hotwallet import HotWallet
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.universe_model import default_universe_options



pytestmark = pytest.mark.skipif(
    (os.environ.get("JSON_RPC_POLYGON") is None) or (shutil.which("anvil") is None),
    reason="Set JSON_RPC_POLYGON env install anvil command to run these tests",
)

@pytest.fixture
def wbtc(web3) -> TokenDetails:
    """Get WBTC on Polygon."""
    return fetch_erc20_details(web3, "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6")


@pytest.fixture
def asset_wbtc(wbtc, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        wbtc.contract.address,
        wbtc.symbol,
        wbtc.decimals,
    )


@pytest.fixture
def wbtc_weth_spot_pair(quickswap_deployment, asset_wbtc, asset_weth) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        asset_wbtc,
        asset_weth,
        "0xdc9232e2df177d7a12fdff6ecbab114e2231198d",
        quickswap_deployment.factory.address,
        fee=0.003,
    )


@pytest.fixture()
def strategy_universe(
    asset_usdc,
    persistent_test_client
) -> TradingStrategyUniverse:
    """Universe that also contains data about our reserve assets."""

    pairs = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.003),
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.polygon, "quickswap", "WETH", "USDC", 0.003),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-10-01"),
        end_at=pd.Timestamp("2023-10-30"),
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_from_dataset(dataset, asset_usdc.address)


def test_generic_routing_multihops_trade(
    web3: Web3,
    hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    generic_routing_model: GenericRouting,
    generic_pricing_model: GenericPricing,
    asset_usdc: AssetIdentifier,
    asset_wbtc: AssetIdentifier,
    wbtc_weth_spot_pair: TradingPairIdentifier,
    weth_usdc_spot_pair: TradingPairIdentifier,
    execution_model: EthereumExecution,
):
    """Open Quickswap position using 3-way trade: USDC -> WETH -> WBTC."""

    routing_model = generic_routing_model

    # Check we have data for both DEXes needed
    exchange_universe = strategy_universe.data_universe.pairs.exchange_universe
    assert exchange_universe.get_exchange_count() == 2
    quickswap = exchange_universe.get_by_chain_and_slug(ChainId.polygon, "quickswap")
    assert quickswap is not None
    
    # Check that our preflight checks pass
    routing_model.perform_preflight_checks_and_logging(strategy_universe.data_universe.pairs)

    sync_model = HotWalletSyncModel(
        web3,
        hot_wallet,
    )

    state = State()
    sync_model.sync_initial(state)

    # Strategy has its reserve balances updated
    sync_model.sync_treasury(datetime.datetime.utcnow(), state, supported_reserves=[asset_usdc])

    assert state.portfolio.get_reserve_position(asset_usdc).quantity == Decimal('10_000')

    # Setup routing state for the approvals of this cycle
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        strategy_universe,
        state,
        generic_pricing_model
    )

    wbtc_price = generic_pricing_model.get_buy_price(
        datetime.datetime.utcnow(),
        wbtc_weth_spot_pair,
        Decimal(500),
    )
    assert wbtc_price.price == pytest.approx(42392.42498236475)

    # Trade on Quickswap spot
    trades = position_manager.open_spot(
        wbtc_weth_spot_pair,
        500.0,
    )
    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )
    assert all([t.is_success() for t in trades])

    assert len(state.portfolio.open_positions) == 1

    # Check our wallet holds all tokens we expect.
    # Note that these are live prices from mainnet,
    # so we do a ranged check.
    balances = fetch_erc20_balances_by_token_list(
        web3,
        hot_wallet.address,
        {
            asset_usdc.address,
            asset_wbtc.address,
        },
        decimalise=True,
    )
    assert 0 < balances[asset_wbtc.address] < 1000, f"Got balance: {balances}"
    assert balances[asset_usdc.address] == pytest.approx(9_500), f"Got balance: {balances}"
