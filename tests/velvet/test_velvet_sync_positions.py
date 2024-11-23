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
from tradeexecutor.strategy.trading_strategy_universe import translate_trading_pair
from tradingstrategy.pair import PandasPairUniverse



def test_check_velvet_universe(
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check we have good universe to map pairs to positions."""
    pair_universe = velvet_test_vault_pair_universe
    for pair in pair_universe.iterate_pairs():
        assert pair.base_token_symbol, f"Got {pair}"
        assert pair.quote_token_symbol, f"Got {pair}"


def test_check_price_on_base_uniswap_v3(
    velvet_test_vault_pricing_model: UniswapV3LivePricing,
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Check that we have a good price feed."""

    pair_universe = velvet_test_vault_pair_universe
    pricing_model = velvet_test_vault_pricing_model

    pair = translate_trading_pair(pair_universe.get_single())

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
        pair_universe=pair_universe,
        vault=base_example_vault,
        hot_wallet=None,
    )

    asset_to_position_map = build_expected_asset_map(
        Portfolio(),
        pair_universe=pair_universe,
        ignore_reserve=True,
    )

    pair = translate_trading_pair(pair_universe.get_single())
    dog_in_me = pair.base

    balances = sync_model.fetch_onchain_balances(
        list(asset_to_position_map.keys())
    )

    balance_map = {a.asset: a for a in balances}

    onchain_balance_data = balance_map[dog_in_me]
    assert onchain_balance_data.amount > 0


def test_velvet_sync_positions(
    base_example_vault: VelvetVault,
    base_usdc: AssetIdentifier,
    velvet_test_vault_pair_universe: PandasPairUniverse,
):
    """Sync velvet open positions

    - Do initial deposit scan.

    - Capture the initial USDC in the vault as a treasury

    - Capture DogMeIn open position
    """

    pair_universe = velvet_test_vault_pair_universe
    assert pair_universe.get_count() == 1

    sync_model = VelvetVaultSyncModel(
        pair_universe=pair_universe,
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

    # Sync DogInMe
    events = sync_model.sync_positions(
        datetime.datetime.utcnow(),
        state,
        pair_universe=pair_universe ,
    )

    # This should have mapped 1 open DogInMe position
    assert len(events) == 1