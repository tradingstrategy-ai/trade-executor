"""Detect missing protocol rules along the metadata-to-execution path.

Tests must start before pair translation: supplying resolved rules directly would
hide a broken loader. Fixed share prices make fees and settlement timing the
only causes of changes in portfolio value and available cash.
"""

import pandas as pd
from tradingstrategy.alternative_data.vault import load_vault_database_with_metadata, convert_vaults_to_trading_pairs
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Use Arbitrum USDC for every vault so fee and settlement assertions do not
#: depend on currency conversion or bridge accounting.
USDC = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
#: Cover immediate execution, event-gated deposits, delayed redemptions, deposit
#: windows, a charged exit and yield parking through the normal metadata loader.
#: Each row is (name, protocol slug, feature flags, address, withdrawal fee fraction).
#: HYPE++ and Atoma need their real addresses to exercise identity-specific rules;
#: other rows use synthetic addresses because their rules depend on features.
#: Leave derived rules out so a missing translation step causes test failures.
VAULTS = (
    ("Generic", "erc-4626", [], "0x" + "1" * 40, 0),
    ("Lagoon", "lagoon-finance", ["lagoon_like", "erc_7540_like"], "0x" + "2" * 40, 0),
    ("Gains", "gains-network", ["gains_like"], "0x" + "3" * 40, 0),
    ("HYPE++", "d2-finance", ["d2_like"], "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004", 0),
    ("Atoma", "atoma", ["atoma_like"], "0x1c788e14d8e5b446e3f71b5142e2edabcab36da1", 0.005),
    ("Cash venue", "fluid", ["fluid_like"], "0x" + "5" * 40, 0),
)


def create_vault_protocol_universe() -> TradingStrategyUniverse:
    """Exercise protocol enrichment without network access or market price changes.

    Pass the declarations through the production metadata parser and pair
    converter so tests detect lost fields at either boundary. Set each share
    price to one USDC so a portfolio loss can be attributed to a fee, and use
    the pairs' common quote asset as the reserve to avoid conversion trades.

    :return:
        Six vaults with daily candles from 1 April to 10 May 2025 inclusive.
        Availability observations are omitted so tests can supply their own
        history or exercise the missing-history policy.
    """
    entries = [dict(chain_id=42161, address=address, name=name, protocol_slug=protocol,
                    features=features, denomination="USDC", normalised_denomination="USDC",
                    denomination_token_address=USDC, denomination_decimals=6,
                    share_token=name, share_token_decimals=18, deposit_fee=0,
                    withdraw_fee=withdrawal, management_fee=0, performance_fee=0,
                    fee_internalised=True)
               for name, protocol, features, address, withdrawal in VAULTS]
    vaults = load_vault_database_with_metadata({"vaults": entries})
    exchanges, pairs = convert_vaults_to_trading_pairs(vaults.export_all_vaults())
    for col in ("token0_decimals", "token1_decimals"):
        pairs[col] = pairs[col].astype('Int64')
    exchange_universe = ExchangeUniverse({e.exchange_id: e for e in exchanges})
    pair_universe = PandasPairUniverse(pairs, exchange_universe=exchange_universe)
    dates = pd.date_range('2025-04-01', '2025-05-10')
    candles = pd.DataFrame([dict(pair_id=p.pair_id, timestamp=d, open=1., high=1., low=1., close=1., volume=0.)
                            for p in pair_universe.iterate_pairs() for d in dates])
    universe = Universe(time_bucket=TimeBucket.d1, chains={ChainId.arbitrum},
                        exchanges=set(exchanges), pairs=pair_universe,
                        candles=GroupedCandleUniverse(candles), liquidity=None)
    result = TradingStrategyUniverse(data_universe=universe, reserve_assets=[])
    result.reserve_assets = [next(result.iterate_pairs()).quote]
    return result
