import datetime

import pandas as pd
import pytest


from eth_defi.token import USDC_NATIVE_TOKEN
from eth_defi.vault.vaultdb import DEFAULT_RAW_PRICE_DATABASE
from tradeexecutor.analysis.vault import display_vaults

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_filter import filter_for_selected_pairs

from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.execution_context import unit_test_execution_context, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data


pytest.mark.skipif(not DEFAULT_RAW_PRICE_DATABASE.exists(), reason="This test requires full vault database bundle in ~/.tradingstrategy/vaults - rsync to get one")


@pytest.fixture()
def universe_options():
    start_at = datetime.datetime(2025, 2, 1)
    end_at = datetime.datetime(2025, 11, 1)

    universe_options = UniverseOptions(
        start_at=start_at,
        end_at=end_at,
    )
    return universe_options



def test_vault_data_has_price_and_tvl(
    persistent_test_client: Client,
    universe_options: UniverseOptions,
):
    """We get vault price and TVL data."""

    client = persistent_test_client

    CHAIN_ID = ChainId.arbitrum
    SUPPORTING_PAIRS = [
        (ChainId.arbitrum, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]
    PREFERRED_STABLECOIN = USDC_NATIVE_TOKEN[CHAIN_ID].lower()

    VAULT_LIST = "0x959f3807f0aa7921e18c78b00b2819ba91e52fef, 0xe5a4f22fcb8893ba0831babf9a15558b5e83446f, 0x75288264fdfea8ce68e6d852696ab1ce2f3e5004, 0x58bfc95a864e18e8f3041d2fcd3418f48393fe6a"

    VAULTS = [
        (ChainId.arbitrum, v.strip()) for v in VAULT_LIST.split(",")
    ]
    execution_context = unit_test_execution_context

    all_pairs_df = client.fetch_pair_universe().to_pandas()
    pairs_df = filter_for_selected_pairs(
        all_pairs_df,
        SUPPORTING_PAIRS,
    )

    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.d1,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        vaults=VAULTS,
        vault_bundled_price_data=DEFAULT_RAW_PRICE_DATABASE,
        check_all_vaults_found=True,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=PREFERRED_STABLECOIN,
        forward_fill=True,
        forward_fill_until=universe_options.end_at,
    )

    assert strategy_universe.has_any_vault_data(), f"Vault data was not loaded properly"

    # Check metadata looks good
    # Plutus HedgeDAO
    vault_address = VAULTS[-1][1]
    vault_pair = strategy_universe.get_pair_by_smart_contract(vault_address)

    assert vault_pair.get_vault_name() == "Plutus Hedge Token"
    assert vault_pair.get_vault_protocol() == "plutus"

    # Check price data looks good
    price_df = strategy_universe.data_universe.candles.get_candles_by_pair(vault_pair.internal_id)
    assert len(price_df) > 0, f"No price data for vault {vault_address}"
    assert  price_df.index.is_monotonic_increasing, "Prices do not look like forward filled time series"

    # Check TVL data looks good
    tvl_df = strategy_universe.data_universe.liquidity.get_samples_by_pair(vault_pair.internal_id)
    assert len(tvl_df) > 0, f"No TVL data for vault {vault_address}"

    assert isinstance(tvl_df.index, pd.MultiIndex)
    assert tvl_df.loc[(vault_pair.internal_id, pd.Timestamp("2025-01-29"))]["open"] == 1
    assert tvl_df.loc[(vault_pair.internal_id, pd.Timestamp("2025-11-26"))]["open"] == pytest.approx(223342.33501)
    assert tvl_df.index.is_monotonic_increasing, "TVLs do not look like forward filled time series"

    # See we can display vault debug data
    display_vaults(
        VAULTS,
        strategy_universe,
        execution_mode=ExecutionMode.unit_testing,
        printer=lambda x: x,
    )