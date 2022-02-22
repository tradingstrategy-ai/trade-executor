import datetime
from decimal import Decimal

from tradingstrategy.pair import PairUniverse, PandasPairUniverse

from eth_hentai.uniswap_v2 import UniswapV2Deployment
from eth_hentai.uniswap_v2_fees import estimate_buy_price_decimals, estimate_sell_price_decimals
from tradeexecutor.state.state import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricingmethod import PricingMethod


class UniswapV2LivePricing(PricingMethod):
    """Always pull the latest prices from Uniswap v2 deployment.

    Currently supports stablecoin pairs only.

    .. note::

        Spot price can be manipulatd - this method is not safe and mostly good
        for testing.

    """

    def __init__(self, uniswap: UniswapV2Deployment, pair_universe: PandasPairUniverse, very_small_amount=Decimal("0.00001")):
        self.uniswap = uniswap
        self.very_small_amount = very_small_amount
        self.pair_universe = pair_universe

    def get_pair(self, pair_id: int):
        return self.pair_universe.get_pair_by_id(pair_id)

    def get_simple_ask_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple buy price without the quantity identified.
        """
        pair = self.get_pair(pair_id)
        assert pair.quote_token_symbol == "USDC"
        return float(estimate_buy_price_decimals(self.uniswap, pair.base_token_address, pair.quote_token_address, self.very_small_amount))

    def get_simple_bid_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple sell price without the quantity identified.
        """
        pair = self.get_pair(pair_id)
        assert pair.quote_token_symbol == "USDC"
        return float(estimate_sell_price_decimals(self.uniswap, pair.base_token_address, pair.quote_token_address, self.very_small_amount))