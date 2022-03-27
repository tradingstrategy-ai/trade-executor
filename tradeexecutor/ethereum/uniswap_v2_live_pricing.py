import logging
import datetime
from decimal import Decimal, ROUND_DOWN

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import TradeExecutorTradingUniverse
from tradingstrategy.pair import PandasPairUniverse, DEXPair

from eth_hentai.token import fetch_erc20_details
from eth_hentai.uniswap_v2.deployment import UniswapV2Deployment
from eth_hentai.uniswap_v2.fees import estimate_buy_price_decimals, estimate_sell_price_decimals
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from web3.exceptions import BadFunctionCallOutput

logger = logging.getLogger(__name__)


class UniswapV2LivePricing(PricingModel):
    """Always pull the latest dollar price for an asset from Uniswap v2 deployment.

    Currently supports stablecoin pairs only.

    .. note::

        Spot price can be manipulatd - this method is not safe and mostly good
        for testing.

    About ask and bid: https://www.investopedia.com/terms/b/bid-and-ask.asp
    """

    def __init__(self,
                 uniswap: UniswapV2Deployment,
                 pair_universe: PandasPairUniverse,
                 very_small_amount=Decimal("0.0001"),
                 supported_stablecoins={"BUSD", "USDC"}):
        self.uniswap = uniswap
        self.web3 = self.uniswap.web3
        self.pair_universe = pair_universe
        self.very_small_amount = very_small_amount
        self.supported_stablecoins = supported_stablecoins

    def get_pair(self, pair_id: int) -> DEXPair:
        return self.pair_universe.get_pair_by_id(pair_id)

    def get_simple_sell_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple sell price without the quantity identified.
        """
        pair = self.get_pair(pair_id)
        assert pair.quote_token_symbol in self.supported_stablecoins, f"The quote token is not dollar like for the {pair}"
        price_for_quantity = estimate_sell_price_decimals(
            self.uniswap,
            self.web3.toChecksumAddress(pair.base_token_address),
            self.web3.toChecksumAddress(pair.quote_token_address),
            self.very_small_amount)
        return float(price_for_quantity / self.very_small_amount)

    def get_simple_buy_price(self, ts: datetime.datetime, pair_id: int) -> USDollarAmount:
        """Get simple buy price without the quantity identified.
        """
        pair = self.get_pair(pair_id)
        logger.info("Getting buy price for %s", pair)
        assert pair.quote_token_symbol in self.supported_stablecoins, f"The quote token is not dollar like for the {pair}"
        try:
            price_for_quantity = estimate_buy_price_decimals(
                self.uniswap,
                self.web3.toChecksumAddress(pair.base_token_address),
                self.web3.toChecksumAddress(pair.quote_token_address),
                self.very_small_amount)
        except BadFunctionCallOutput as e:
            # TODO: Ganache hack
            raise RuntimeError(f"Could not get price for %s, %s", pair, pair.address)

        return float(price_for_quantity / self.very_small_amount)

    def quantize_quantity(self, pair_id: int, quantity: float, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        pair = self.get_pair(pair_id)
        base_details = fetch_erc20_details(self.uniswap.web3, self.web3.toChecksumAddress(pair.base_token_address))
        decimals = base_details.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=ROUND_DOWN)


def uniswap_v2_live_pricing_factory(execution_model: ExecutionModel, universe: TradeExecutorTradingUniverse) -> UniswapV2LivePricing:
    assert isinstance(execution_model, UniswapV2ExecutionModel), "Pricing method is not compatible with this execution model"
    assert isinstance(universe, TradingStrategyUniverse), f"This pricing method only works with TradingStrategyUniverse, we received {universe}"
    uniswap = execution_model.uniswap
    universe = universe.universe
    return UniswapV2LivePricing(uniswap, universe.pairs)
