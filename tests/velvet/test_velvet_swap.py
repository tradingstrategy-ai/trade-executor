"""Execute trades using Velvet vault and Enso."""
import datetime
from decimal import Decimal

import pytest
from eth_defi.token import TokenDetails


from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse



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
):
    """Perform an intent-based swap

    - Do initial deposit scan. Capture the initial USDC and DogInMe in the vault.

    - Prepare a trade execution

    - Execute it

    - See we got new tokens
    """

    state = state_with_starting_positions
    strategy_universe = velvet_test_vault_strategy_universe

    position_manager = PositionManager(
        datetime.datetime.utcnow(),
        universe=velvet_test_vault_strategy_universe,
        state=state,
        pricing_model=velvet_pricing_model,
        default_slippage_tolerance=0.05,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v2", "SKI", "WETH"))

    position_manager.open_spot(
        pair=pair,
        value=1.0  # USDC
    )


