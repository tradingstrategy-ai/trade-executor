"""Execute trades using Velvet vault and Enso."""
import datetime
import os
from decimal import Decimal

import pytest
from eth_defi.token import TokenDetails


from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.velvet.execution import VelvetExecution
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


pytestmark = pytest.mark.skipif(
    not os.environ.get("VELVET_VAULT_OWNER_PRIVATE_KEY"),
    reason="Need to set VELVET_VAULT_OWNER_PRIVATE_KEY to a specific private key to run this test"
)


@pytest.fixture()
def state_with_starting_positions(
    velvet_test_vault_strategy_universe,
    velvet_pricing_model,
    base_example_vault,
    base_usdc,
) -> State:
    """Scan the initial positions for the tests."""

    strategy_universe = velvet_test_vault_strategy_universe
    pricing_model = velvet_pricing_model

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Get the initial positions
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    sync_model.sync_treasury(
        datetime.datetime.utcnow(),
        state,
        supported_reserves=state.portfolio.get_reserve_assets(),
    )
    sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )

    # We have DogInMe position + cash
    assert len(portfolio.open_positions) == 1
    assert len(state.portfolio.get_reserve_assets()) == 1
    assert state.portfolio.get_cash() == pytest.approx(2.674828)
    assert portfolio.open_positions[1].get_quantity() == pytest.approx(Decimal(580.917745152826802993))

    return state


def test_velvet_intent_based_open_position_uniswap_v2(
    web3: Web3,
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    base_doginme: AssetIdentifier,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: UniswapV2LivePricing,
    base_usdc_token: TokenDetails,
    state_with_starting_positions: State,
    velvet_execution_model: VelvetExecution,
    velvet_routing_model: VelvetEnsoRouting,
):
    """Perform an intent-based swap

    - Do initial deposit scan. Capture the initial USDC and DogInMe in the vault.

    - Prepare a trade execution

    - Execute it

    - See we got new tokens
    """

    state = state_with_starting_positions
    strategy_universe = velvet_test_vault_strategy_universe
    execution_model = velvet_execution_model
    routing_model = velvet_routing_model

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=velvet_test_vault_strategy_universe,
        state=state,
        pricing_model=velvet_pricing_model,
        default_slippage_tolerance=0.05,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v2", "SKI", "WETH"))

    trades = position_manager.open_spot(
        pair=pair,
        value=1.0  # USDC
    )

    # Setup routing state for the approvals of this cycle
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)

    execution_model.initialize()

    execution_model.execute_trades(
        datetime.datetime.utcnow(),
        state,
        trades,
        routing_model,
        routing_state,
        check_balances=True,
    )