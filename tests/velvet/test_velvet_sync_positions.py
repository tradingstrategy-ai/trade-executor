"""Test Velvet sync model."""
import datetime

import pytest

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import UniswapV3LivePricing
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.portfolio import Portfolio

from tradeexecutor.state.state import State
from tradeexecutor.strategy.asset import build_expected_asset_map
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair, TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse


def test_check_velvet_universe(
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check we have good universe to map pairs to positions."""
    pair_universe = velvet_test_vault_pair_universe

    assert pair_universe.get_count() == 2

    for pair in pair_universe.iterate_pairs():
        assert pair.base_token_symbol, f"Got {pair}"
        assert pair.quote_token_symbol, f"Got {pair}"


def test_check_price_on_base_uniswap_v3(
    velvet_test_vault_pricing_model: UniswapV3LivePricing,
    velvet_test_vault_strategy_universe: PandasPairUniverse,
):
    """Check that we have a good price feed."""

    strategy_universe = velvet_test_vault_strategy_universe
    pricing_model = velvet_test_vault_pricing_model

    pair_desc = (ChainId.base, "uniswap-v3", "DogInMe", "WETH", 0.01)

    pair = strategy_universe.get_pair_by_human_description(pair_desc)

    # Read DogMeIn/USDC pool price
    # Note this pool lacks liquidity, so we do not care about the value
    mid_price = pricing_model.get_mid_price(
        datetime.datetime.utcnow(),
        pair,
    )

    assert mid_price > 0


def test_velvet_fetch_balances(
    base_example_vault: VelvetVault,
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check we fetch balances for onchain positions correctly.

    - Read DogInMe balance onchain
    """

    pair_universe = velvet_test_vault_pair_universe

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    asset_to_position_map = build_expected_asset_map(
        Portfolio(),
        pair_universe=pair_universe,
        ignore_reserve=True,
    )

    pair_desc = (ChainId.base, "uniswap-v3", "DogInMe", "WETH", 0.01)
    pair = translate_trading_pair(
        pair_universe.get_pair_by_human_description(pair_desc),
    )
    dog_in_me = pair.base

    balances = sync_model.fetch_onchain_balances(
        list(asset_to_position_map.keys())
    )

    balance_map = {a.asset: a for a in balances}

    onchain_balance_data = balance_map[dog_in_me]
    assert onchain_balance_data.amount > 0


def test_velvet_sync_positions_initial(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_test_vault_pricing_model: UniswapV3LivePricing,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury

    - Capture DogMeIn open position
    """

    strategy_universe = velvet_test_vault_strategy_universe
    pair_universe = strategy_universe.data_universe.pairs
    pricing_model = velvet_test_vault_pricing_model
    assert pair_universe.get_count() == 2

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Sync USDC
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    cycle = datetime.datetime.utcnow()
    sync_model.sync_treasury(cycle, state)
    assert portfolio.get_cash() == pytest.approx(2.674828)

    # Sync DogInMe - creates initial position no event generated
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )
    assert len(events) == 0

    # We have DogInMe position
    assert len(portfolio.open_positions) == 1
    dog_in_me_pos =  portfolio.open_positions[1]
    assert dog_in_me_pos.get_value() > 0



def test_velvet_sync_positions_deposit(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    velvet_test_vault_strategy_universe: TradingStrategyUniverse,
    velvet_test_vault_pricing_model: UniswapV3LivePricing,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury

    - Capture DogMeIn open position
    """

    strategy_universe = velvet_test_vault_strategy_universe
    pair_universe = strategy_universe.data_universe.pairs
    pricing_model = velvet_test_vault_pricing_model

    sync_model = VelvetVaultSyncModel(
        vault=base_example_vault,
        hot_wallet=None,
    )

    state = State()
    portfolio = state.portfolio

    # Sync USDC
    sync_model.sync_initial(
        state,
        reserve_asset=base_usdc,
        reserve_token_price=1.0,
    )
    # Sync DogInMe - creates initial position no event generated
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
    )

    # We have DogInMe position + cash
    assert len(portfolio.open_positions) == 1
