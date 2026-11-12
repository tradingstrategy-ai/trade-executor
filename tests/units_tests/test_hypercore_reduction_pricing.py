"""HyperCore reduction pricing keeps the gross mid so vault fees are charged once.

Why: ``PositionManager.adjust_position()`` prices a HyperCore partial sell at its
implied share value. It used to set the trade's mid price to that value as well.
In backtests the pricing model quotes vault sales net of exit-time fees, and
execution recalculates those fees from ``planned_mid_price``, so every HyperCore
trim was charged its performance and exit fees twice (Hyper-AI research backtest,
2026-07-12: a Realist Capital trim planned at $19,966.87 net realised $19,273.42).
"""
from decimal import Decimal

import pytest

from tradeexecutor.strategy.pandas_trader.position_manager import price_hypercore_reduction
from tradeexecutor.strategy.trade_pricing import TradePricing


def test_hypercore_reduction_keeps_gross_mid():
    """Price HyperCore reductions at the implied share value and keep the quote's fee ratio.

    1. Price the 2026-07-12 Realist Capital trim, 2,661.63 shares marked at $19,966.87 net
       and quoted 7.5017 net against a 7.7915 gross mid
    2. Check the planned price is the implied net share value and the planned mid stays gross,
       so execution's fee recalculation starts from the gross price and charges the fees once
    3. Price a quote without a fee haircut, as live HyperCore quotes are
    4. Check price and mid both equal the implied share value, as before the fix
    5. Check a purchase is rejected
    """
    # 1. Price the 2026-07-12 Realist Capital trim, 2,661.63 shares marked at $19,966.87 net
    #    and quoted 7.5017 net against a 7.7915 gross mid
    quantity = 2661.629987
    fee_net_quote = TradePricing(price=7.501744455620538, mid_price=7.791519546751047, lp_fee=0.0, pair_fee=0.0)
    priced = price_hypercore_reduction(fee_net_quote, -19966.86799789, -quantity)

    # 2. Check the planned price is the implied net share value and the planned mid stays gross,
    #    so execution's fee recalculation starts from the gross price and charges the fees once
    assert priced.price == pytest.approx(19966.86799789 / quantity)
    assert priced.mid_price == pytest.approx(7.791519546751047, rel=1e-9)
    assert priced.mid_price / priced.price == pytest.approx(fee_net_quote.mid_price / fee_net_quote.price)

    # 3. Price a quote without a fee haircut, as live HyperCore quotes are
    live_quote = TradePricing(price=1.0, mid_price=1.0, lp_fee=0.0, pair_fee=0.0)
    priced = price_hypercore_reduction(live_quote, -1234.5, Decimal("-100"))

    # 4. Check price and mid both equal the implied share value, as before the fix
    assert priced.price == pytest.approx(12.345)
    assert priced.mid_price == pytest.approx(12.345)

    # 5. Check a purchase is rejected
    with pytest.raises(AssertionError):
        price_hypercore_reduction(live_quote, 100.0, -1.0)
