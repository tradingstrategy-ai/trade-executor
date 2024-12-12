"""Execute trades using Velvet vault and Enso."""
import datetime
import os
from decimal import Decimal

import pytest

from tradeexecutor.ethereum.velvet.execution import VelvetExecution
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradingstrategy.chain import ChainId

from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Detect Github Actions
CI = os.environ.get("CI", None) is not None


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


def test_keycat_weth_price(
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: GenericPricing,
):
    """See we correctly quote KEYCAT/WETH pair price in USDC.

    - We want all prices quoted in dollars
    """
    strategy_universe = velvet_test_vault_strategy_universe
    pricing_model = velvet_pricing_model

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v2", "KEYCAT", "WETH"))
    weth = pair.quote

    weth_price = pricing_model.get_exchange_rate(
        datetime.datetime.utcnow(),
        weth,
    )

    assert 2000 < weth_price < 10_000

    price = pricing_model.get_mid_price(datetime.datetime.utcnow(), pair)
    assert 0.001 < price < 0.3

    price_structure = pricing_model.get_buy_price(datetime.datetime.utcnow(), pair, Decimal(1.0))
    assert 0.001 < price_structure.price < 0.3

    price_structure = pricing_model.get_sell_price(datetime.datetime.utcnow(), pair, Decimal(1.0))
    assert 0.001 < price_structure.price < 0.3


@pytest.mark.skipif(CI, reason="Skipped on continuous integration due to Velvet/Enso stability issues")
def test_velvet_intent_based_open_position_uniswap_v2(
    state_with_starting_positions: State,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: GenericPricing,
    velvet_execution_model: VelvetExecution,
    velvet_routing_model: VelvetEnsoRouting,
):
    """Perform an intent-based swap

    - Do initial deposit scan. Capture the initial USDC and DogInMe in the vault.

    - Prepare a trade execution on KEYCAT Uniswap v2

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
        default_slippage_tolerance=0.20,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v2", "KEYCAT", "WETH"))

    trades = position_manager.open_spot(
        pair=pair,
        value=1.0  # USDC
    )
    t = trades[0]

    assert t.planned_reserve == pytest.approx(Decimal(1.0))

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

    assert t.is_success(), f"Enso trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert 0 < t.executed_price < 1
    assert t.executed_quantity > 10  # 129 as writing of this
    assert t.executed_reserve == pytest.approx(Decimal(1))



def test_velvet_intent_based_reduce_position_uniswap_v3(
    state_with_starting_positions: State,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: GenericPricing,
    velvet_execution_model: VelvetExecution,
    velvet_routing_model: VelvetEnsoRouting,
):
    """Perform an intent-based swap

    - Do initial deposit scan. Capture the initial USDC and DogInMe in the vault.

    - Prepare sell existing DogInMe position partially

    - Execute it
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
        default_slippage_tolerance=0.20,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v3", "DogInMe", "WETH"))
    position = position_manager.get_current_position_for_pair(pair)
    quantity_to_reduce = position.get_quantity() / Decimal(2)
    dollar_delta = position.get_value() * float(quantity_to_reduce / position.get_quantity())
    trades = position_manager.adjust_position(
        pair=pair,
        dollar_delta=-dollar_delta,
        quantity_delta=-quantity_to_reduce,
        weight=1,
    )
    t = trades[0]
    assert t.is_sell()
    assert 0 < t.planned_reserve < 1
    assert t.planned_quantity == pytest.approx(Decimal(-290.458872576413401496))
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

    assert t.is_success(), f"Enso trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert 0 < t.executed_price < 1
    assert t.executed_quantity == pytest.approx(Decimal('-290.458872576413401496'))
    assert t.executed_reserve > 0


def test_velvet_intent_based_close_position_uniswap_v3(
    state_with_starting_positions: State,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: GenericPricing,
    velvet_execution_model: VelvetExecution,
    velvet_routing_model: VelvetEnsoRouting,
):
    """Perform an intent-based swap

    - Close exiting DogInMe positio fully
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
        default_slippage_tolerance=0.20,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v3", "DogInMe", "WETH"))
    position = position_manager.get_current_position_for_pair(pair)
    trades = position_manager.close_position(position)
    t = trades[0]
    assert t.is_sell()
    assert 0 < t.planned_reserve < 1
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

    assert t.is_success(), f"Enso trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert 0 < t.executed_price < 1
    assert t.executed_quantity == pytest.approx(Decimal('-580.917745152826802993'))
    assert t.executed_reserve > 0

    assert len(state.portfolio.closed_positions) == 1


def test_velvet_intent_based_increase_position_uniswap_v3(
    state_with_starting_positions: State,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_pricing_model: GenericPricing,
    velvet_execution_model: VelvetExecution,
    velvet_routing_model: VelvetEnsoRouting,
):
    """Perform an intent-based swap

    - Do initial deposit scan. Capture the initial USDC and DogInMe in the vault.

    - Prepare sell existing DogInMe position partially

    - Execute it
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
        default_slippage_tolerance=0.25,
    )

    pair = strategy_universe.get_pair_by_human_description((ChainId.base, "uniswap-v3", "DogInMe", "WETH"))
    dollar_delta = 0.1  # Buy 0.1 more on the existing position
    position = position_manager.get_current_position_for_pair(pair)
    quantity_delta = dollar_delta * position.get_current_price()
    trades = position_manager.adjust_position(
        pair=pair,
        dollar_delta=dollar_delta,
        quantity_delta=quantity_delta,
        weight=1,
    )
    t = trades[0]
    assert t.is_buy()
    assert 0 < t.planned_reserve < 1
    assert t.planned_quantity > 100
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

    assert t.is_success(), f"Enso trade failed: {t.blockchain_transactions[0].revert_reason}"
    assert 0 < t.executed_price < 1
    assert t.executed_quantity > 100
    assert t.executed_reserve > 0