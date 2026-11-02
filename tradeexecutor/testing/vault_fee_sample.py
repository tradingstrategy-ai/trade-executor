"""Build an offline vault backtest from unmodified metadata and observed prices."""

import json
from pathlib import Path

import pandas as pd
from tradingstrategy.alternative_data.vault import load_vault_database_with_metadata, convert_vaults_to_trading_pairs
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.dex_data_translation import translate_trading_pair
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def load_fee_sample(root: Path, protocol: str) -> TradingStrategyUniverse:
    """Load a real vault's identity, rates and prices via the normal translation path."""
    data = json.loads((root / "metadata.json").read_text())
    data['vaults'] = [r for r in data['vaults'] if protocol in r['protocol_slug']]
    vaults = load_vault_database_with_metadata(data)
    exchanges, pairs = convert_vaults_to_trading_pairs(vaults.export_all_vaults())
    # JSON decimal counts can be float; restore the actual integer token precision.
    for column in ('token0_decimals', 'token1_decimals'):
        pairs[column] = pairs[column].astype('Int64')
    exchange_universe = ExchangeUniverse({e.exchange_id: e for e in exchanges})
    pair_universe = PandasPairUniverse(pairs, exchange_universe=exchange_universe)
    pair = translate_trading_pair(next(pair_universe.iterate_pairs()))
    row = data['vaults'][0]
    prices = pd.read_csv(root / 'prices.csv', parse_dates=['timestamp'])
    prices = prices.loc[prices.address.eq(row['address'])].set_index('timestamp').share_price.resample('1D').last().ffill()
    candles = pd.DataFrame({'timestamp': prices.index, 'open': prices.values, 'high': prices.values,
                            'low': prices.values, 'close': prices.values, 'volume': 0., 'pair_id': pair.internal_id})
    universe = Universe(time_bucket=TimeBucket.d1, chains={pair_universe.get_single().chain_id},
                        exchanges=set(exchanges), pairs=pair_universe,
                        candles=GroupedCandleUniverse(candles), liquidity=None)
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[pair.quote])


class PendingFeeSnapshot(ExecutionTestHook):
    """Capture an unresolved withdrawal and its wallet for an isolated restart check."""

    def __init__(self):
        self.state_json = None
        self.balances = None

    def on_before_cycle(self, cycle, cycle_st, state, sync_model):
        if self.state_json is None and any(t.is_sell() and t.get_status() == TradeStatus.vault_settlement_pending
                                           for t in state.portfolio.get_all_trades()):
            self.state_json = state.to_json()
            self.balances = dict(sync_model.wallet.balances)
