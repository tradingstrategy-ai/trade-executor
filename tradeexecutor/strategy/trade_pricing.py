"""Trade pricing and price impact."""

import datetime
from _decimal import Decimal
from logging import getLogger
from dataclasses import dataclass
from typing import Optional, List

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount, BPS, USDollarPrice
from dataclasses_json import dataclass_json

from tradingstrategy.types import Percent

logger = getLogger(__name__)


class PriceImpactToleranceExceeded(Exception):
    """Crash the executor if we try accidentally pass in a trade with too much price impact.

    Layers of price impact protection

    - :py:class:`SizeRiskModel` estimates the cap of the trade,
       resizes trades approriately

    - :py:class:`PriceImpactToleranceExceeded` crashes the trade executor
      if it detects a trade with too much impact going to the execution

    - Enzyme smart contracts have CumulativeSlippageTolerance of 10% per week

    - See :py:func:`post_process_trade_decision`
    """



@dataclass_json
@dataclass(slots=True, frozen=True)
class TradePricing:
    """Describe price results for a price query.

    - Each price result is tied to quantiy/amount

    - Each price result gets a split that describes liquidity provider fees

    A helper class to deal with problems of accounting and estimation of prices on Uniswap like exchange.
    """

    #: The price we expect this transaction to clear.
    #:
    #: - LP fees included
    #: - Price impact included
    #: - Slippage = 0
    #:
    price: USDollarPrice

    #: The theoretical market price during the transaction.
    #:
    #: This is the `(ask price + bid price) / 2` order book price
    #: that no one can obtain.
    #:
    #: No LP fees, price impact, etc. are included in this price.
    #: It can be used as a basis for other fee estimation
    #: calculations.
    #:
    #: See :term:`mid price` for more information.
    #:
    mid_price: USDollarPrice

    #: How much liquidity provider fees we are going to pay on this trade.
    #:
    #: Set to None if data is not available.
    #:
    #: Each trading pair on path will have its own fees.
    #: The list is the per path fees.
    #:
    lp_fee: Optional[list[USDollarAmount]] = None

    #: How much token tax was paid
    #:
    #: lp_fee is inclusive of token_tax.
    token_tax: Optional[USDollarAmount] = None

    token_tax_percent: Optional[Percent] = None

    #: What was the LP fee % used as the base of the calculations.
    #:
    pair_fee: Optional[list[BPS]] = None

    #: How old price data we used for this estimate
    #:
    market_feed_delay: Optional[datetime.timedelta] = None

    #: Is this buy or sell trade.
    #:
    #:
    #: True for buy.
    #: False for sell.
    #: None for Unknown.
    side: Optional[bool] = None
    
    #: Path of the trade
    #: One trade can have multiple swaps if there is an intermediary pair.
    path: Optional[List[TradingPairIdentifier]] = None

    #: When the price read was performed
    read_at: Optional[datetime.datetime] = None

    #: What was the block number when the read was performed
    block_number: Optional[int] = None

    #: Amount of tokens we put in
    #:
    #: Tracked for debugging
    #:
    token_in: Optional[Decimal] = None

    #: Amount of tokens we got out
    #:
    #: Tracked for debugging
    #:
    token_out: Optional[Decimal] = None

    def __repr__(self):
        fee_list = [fee or 0 for fee in self.pair_fee]
        if self.block_number:
            block_text = f"block:{self.block_number:,}"
        else:
            block_text = ""
        return f"<TradePricing:{self.price} mid:{self.mid_price} fee:{format_fees_percentage(fee_list)} {block_text}>"
    
    def __post_init__(self):
        """Validate parameters.

        Make sure we don't slip in e.g. NumPy types.
        """
        assert type(self.price) == float, f"Expected price to be float, got {self.price} {type(self.price)}"
        assert type(self.mid_price) == float

        # Convert legacy single lp_fee model to path based model
        if type(self.lp_fee) != list:
            object.__setattr__(self, 'lp_fee', [self.lp_fee])
        
        if type(self.pair_fee) != list:
            object.__setattr__(self, 'pair_fee', [self.pair_fee])
        
        assert all([type(_lp_fee) in {float, type(None)} for _lp_fee in self.lp_fee]), f"lp_fee must be provided as type list with float or NoneType elements. Got Got lp_fee: {self.lp_fee} {type(self.lp_fee)}"
        
        assert all([type(_pair_fee) in {float, type(None)} for _pair_fee in self.pair_fee]), f"pair_fee must be provided as a list with float or NoneType elements. Got fee: {self.pair_fee} {type(self.pair_fee)} "
        
        if self.market_feed_delay is not None:
            assert isinstance(self.market_feed_delay, datetime.timedelta)

        # Do sanity checks for the price calculation, e.g. in the case there has been a negative price somewhere
        if self.side is not None:
            if self.side:
                assert self.price >= self.mid_price, f"Got bad buy pricing: {self.price} > {self.mid_price}"
            if not self.side:
                assert self.price <= self.mid_price, f"Got bad sell pricing: {self.price} < {self.mid_price}"
                
        if self.path:
            assert all([type(address) == TradingPairIdentifier for address in self.path]), "path must be provided as a list of TradePairIdentifier"

    def get_total_lp_fees(self) -> USDollarAmount:
        """Returns the total lp fees paid (dollars) for the trade."""
        if all(self.lp_fee):
            return sum(self.lp_fee)
        
        # logger.warning("some trades don't have an associated lp fee")
        
        return sum(filter(None, self.lp_fee))
    
    def get_fee_percentage(self):
        """Returns a single decimal value for the percentage of fees paid. 
        This calculation represents the average of all the pair fees. 
        Calculation is the same for v2 and v3.
        
        Calculation:
        
        -> x(1 - fee0)(1 - fee1) = x(1 - fee)
        -> (1 - fee0)(1 - fee1) = (1 - fee)
        -> fee = 1 - (1 - fee0)(1 - fee1)        
        """

        if all(self.pair_fee):
            if len(self.pair_fee) == 1:
                assert self.pair_fee[0] < 1
                return self.pair_fee[0]
            elif len(self.pair_fee) == 2:
                assert all([0 <= fee < 1 for fee in self.pair_fee])
                return 1 - (1 - self.pair_fee[0]) * (1 - self.pair_fee[1])
            else:
                raise ValueError("Swap involves fees from more than two pairs")
            
        # logger.warning("some pairs in the trade have a fee of None")
        return None

    def get_price_impact(self) -> Percent:
        """How far off we are from the mid-price with this trade."""
        return abs((self.price - self.mid_price) / self.mid_price)


def format_fees_percentage(fees: list[BPS]) -> str:
    """Returns string of formatted fees
    
    e.g. fees = [0.03, 0.005]
    => 0.3000% 0.0500%
    
    :param fees:
        list of lp fees in float (multiplier) format
        
    :returns:
        formatted str
    """
    _fees = [fee or 0 for fee in fees]
    strFormat = len(_fees) * '{:.4f}% '
    return strFormat.format(*_fees)
    
    
def format_fees_dollars(fees: list[USDollarAmount] | USDollarAmount) -> str:
    """Returns string of formatted fees
    
    :param fees:
        Can either be a list of fees or a single fee
    
    e.g. fees = [30, 50]
    => $30.00 $50.00
    
    :param fees:
        list of fees paid in absolute value (dollars)
    
    :returns:
        formatted str
    """
    
    if type(fees) != list:
        return f"${fees:.2f}"
    
    _fees = [fee or 0 for fee in fees]
    strFormat = len(_fees) * '${:.2f} '
    return strFormat.format(*_fees)