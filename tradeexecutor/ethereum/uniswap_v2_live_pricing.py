import datetime
from decimal import Decimal

from eth_hentai.uniswap_v2 import estimate_sell_price_decimals, UniswapV2Deployment
from eth_hentai.uniswap_v2_fees import estimate_buy_price_decimals
from tradeexecutor.state.state import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricingmethod import PricingMethod


class UniswapV2LivePricing(PricingMethod):
    """Always pull the latest prices from Uniswap v2 deployment.

    Currently supports stablecoin pairs only.
    """

    def __init__(self, uniswap: UniswapV2Deployment, very_small_amount=Decimal("0.00001")):
        self.uniswap = uniswap
        self.very_small_amount = very_small_amount

    def get_simple_ask_price(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Get simple buy price without the quantity identified.
        """
        assert pair.quote.token_symbol == "USDC"
        return float(estimate_buy_price_decimals(self.uniswap, pair.base.address, pair.quote.address, 1.0))


    def get_simple_bid_price(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> USDollarAmount:
        """Get simple sell price without the quantity identified.
        """
        assert pair.quote.token_symbol == "USDC"
        return float(estimate_sell_price_decimals(self.uniswap, pair.base.address, pair.quote.address, self.very_small_amount))