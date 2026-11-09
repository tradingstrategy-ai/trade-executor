"""Hardcoded raw metadata universe exercising normal vault loading and translation."""

import pandas as pd
from tradingstrategy.alternative_data.vault import load_vault_database_with_metadata, convert_vaults_to_trading_pairs
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


#: Reserve denomination used by every test venue, avoiding unrelated bridge transport.
USDC = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
#: Declarations are raw input, not pre-enriched pair fields.
VAULTS = (
    ("Generic", "erc-4626", [], "0x" + "1" * 40, 0),
    ("Lagoon", "lagoon-finance", ["lagoon_like", "erc_7540_like"], "0x" + "2" * 40, 0),
    ("Gains", "gains-network", ["gains_like"], "0x" + "3" * 40, 0),
    ("HYPE++", "d2-finance", ["d2_like"], "0x75288264fdfea8ce68e6d852696ab1ce2f3e5004", 0),
    ("Atoma", "atoma", ["atoma_like"], "0x1c788e14d8e5b446e3f71b5142e2edabcab36da1", 0.005),
    ("Cash venue", "fluid", ["fluid_like"], "0x" + "5" * 40, 0),
)


def create_vault_protocol_universe() -> TradingStrategyUniverse:
    """Construct a fixed-price universe through real parser and lazy pair conversion.

    Raw availability samples pass through the real state converter. No protocol
    rule, queue classification or tax is injected by this helper.
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
