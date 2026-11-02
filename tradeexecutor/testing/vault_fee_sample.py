"""Build an offline vault backtest from unmodified metadata and observed prices."""

import datetime
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
from tradingstrategy.alternative_data.vault import load_vault_database_with_metadata, convert_vaults_to_trading_pairs
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.dex_data_translation import translate_trading_pair
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def load_fee_sample(root: Path, protocol: str) -> TradingStrategyUniverse:
    """Build an offline, single-vault universe for fee integration tests.

    Use normal metadata translation so fee detection exercises the same path as
    downloaded data. Observed share prices become daily OHLC candles using the
    last observation and forward-filling gaps; this is a test sampling assumption.

    :param root:
        Fixture directory containing ``metadata.json`` and Zstandard-compressed
        ``prices.parquet``. Prices need naive UTC ``timestamp``, ``address`` and
        denomination-token ``share_price`` columns.
    :param protocol:
        Protocol-slug substring selecting exactly one vault in this fixture.
    :return:
        Daily single-vault universe whose reserve asset is its denomination
        token. No network request or synthetic fee adjustment is made.
    """
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
    prices = pd.read_parquet(root / 'prices.parquet')
    prices = prices.loc[prices.address.eq(row['address'])].set_index('timestamp').share_price.resample('1D').last().ffill()
    candles = pd.DataFrame({'timestamp': prices.index, 'open': prices.values, 'high': prices.values,
                            'low': prices.values, 'close': prices.values, 'volume': 0., 'pair_id': pair.internal_id})
    universe = Universe(time_bucket=TimeBucket.d1, chains={pair_universe.get_single().chain_id},
                        exchanges=set(exchanges), pairs=pair_universe,
                        candles=GroupedCandleUniverse(candles), liquidity=None)
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[pair.quote])


class PendingFeeSnapshot(ExecutionTestHook):
    """Capture an unresolved withdrawal and its wallet for an isolated restart check."""

    def __init__(self) -> None:
        """Start without a capture; later cycles must not overwrite the first one."""
        self.state_json: str | None = None
        self.balances: dict[str, Decimal] | None = None

    def on_before_cycle(self, cycle: int, cycle_st: datetime.datetime,
                        state: State, sync_model: BacktestSyncModel) -> None:
        """Capture normal serialised state and balances while an exit is pending.

        The restart test restores these into fresh objects to check that request
        fee inputs survive and a repeated resolver call cannot credit cash twice.
        ``cycle`` and ``cycle_st`` are unused hook-interface arguments: capture
        follows trade status rather than a hard-coded schedule.

        :param state:
            Current backtest state, serialised once at the first pending sell.
        :param sync_model:
            Backtest treasury model whose simulated wallet balances are copied.
        """
        if self.state_json is None and any(t.is_sell() and t.get_status() == TradeStatus.vault_settlement_pending
                                           for t in state.portfolio.get_all_trades()):
            self.state_json = state.to_json()
            self.balances = dict(sync_model.wallet.balances)
